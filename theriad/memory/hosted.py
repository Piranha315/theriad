"""Supabase-backed implementation of the user memory store.

It mirrors the small ``MemoryStore`` surface used by the API and prompt
builder, but never creates ``data/memory-users`` files. Imported collections
and model-memory policies are kept in tenant-scoped Postgres rows.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from theriad.memory.store import MemoryEntry, _normalise_model_id, auto_tags
from theriad.storage.hosted import (
    HostedStorageUnavailable,
    MemoryProfileRepository,
    MemoryRepository,
    SupabaseRestClient,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HostedMemoryStore:
    """MemoryStore-compatible tenant repository."""

    def __init__(self, organization_id: str, user_id: str, client: SupabaseRestClient | None = None):
        if not organization_id or not user_id:
            raise HostedStorageUnavailable("Hosted memory requires an authenticated organization.")
        self.organization_id = organization_id
        self.user_id = user_id
        self.client = client or SupabaseRestClient()
        self.entries = MemoryRepository(self.client)
        self.profiles = MemoryProfileRepository(self.client)

    @property
    def path(self):
        """Compatibility label; no local filesystem path exists in hosted mode."""
        return f"supabase://memory/{self.organization_id}/{self.user_id}"

    @property
    def name(self) -> str:
        return "memory"

    def set_name(self, new_name: str) -> None:
        # Names are a local UX alias, not a separate hosted resource. Validate
        # through the same helper but keep the hosted schema minimal for now.
        from theriad.memory.store import _normalise_alias

        _normalise_alias(new_name)

    def _rows(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self.entries.list_for_owner(
            organization_id=self.organization_id,
            user_id=self.user_id,
            order="updated_at.desc",
            limit=limit,
        )

    @staticmethod
    def _entry(row: dict[str, Any]) -> MemoryEntry:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        return MemoryEntry(
            id=str(row.get("id") or uuid.uuid4()),
            text=str(row.get("content") or ""),
            title=str(row.get("label") or "").strip() or None,
            tags=[str(tag) for tag in metadata.get("tags", []) if str(tag).strip()],
            source=str(row.get("source") or "manual"),
            conversation_title=metadata.get("conversation_title"),
            upload_id=metadata.get("upload_id"),
            upload_name=metadata.get("upload_name"),
            model_id=row.get("model_id"),
            created_at=str(row.get("created_at") or _now()),
            updated_at=str(row.get("updated_at") or row.get("created_at") or _now()),
        )

    @staticmethod
    def _metadata(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "tags": list(entry.tags),
            "conversation_title": entry.conversation_title,
            "upload_id": entry.upload_id,
            "upload_name": entry.upload_name,
        }

    def list(self, *, source: str | None = None, tag: str | None = None, query: str | None = None):
        entries = [self._entry(row) for row in self._rows()]
        if source:
            entries = [entry for entry in entries if entry.source == source.lower()]
        if tag:
            entries = [entry for entry in entries if tag.lower() in entry.tags]
        if query:
            lowered = query.lower()
            entries = [
                entry for entry in entries
                if lowered in entry.text.lower()
                or lowered in (entry.title or "").lower()
                or lowered in (entry.conversation_title or "").lower()
                or any(lowered in value.lower() for value in entry.tags)
            ]
        return entries

    def get(self, entry_id: str) -> MemoryEntry | None:
        row = self.entries.get_owned(
            entry_id, organization_id=self.organization_id, user_id=self.user_id
        )
        return self._entry(row) if row else None

    def add(
        self,
        text: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        conversation_title: str | None = None,
    ) -> MemoryEntry:
        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("memory text must not be empty")
        entry = MemoryEntry(
            text=cleaned,
            title=title.strip() or None if title is not None else None,
            tags=[tag.strip().lower() for tag in (tags or []) if tag.strip()] or auto_tags(cleaned),
            source=source.lower(),
            conversation_title=conversation_title,
        )
        self.entries.insert_owned(
            {
                "id": entry.id,
                "project_id": None,
                "model_id": None,
                "label": entry.title or "Memory",
                "content": entry.text,
                "source": entry.source,
                "metadata": self._metadata(entry),
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
            },
            organization_id=self.organization_id,
            user_id=self.user_id,
        )
        return entry

    def update(self, entry_id: str, *, text=None, title=None, tags=None, source=None):
        current = self.get(entry_id)
        if current is None:
            return None
        if text is not None and text.strip():
            current.text = text.strip()
        if title is not None:
            current.title = title.strip() or None
        if tags is not None:
            current.tags = [tag.strip().lower() for tag in tags if tag.strip()]
        if source is not None:
            current.source = source.lower()
        current.updated_at = _now()
        self.entries.update_owned(
            entry_id,
            {
                "content": current.text,
                "label": current.title or "Memory",
                "source": current.source,
                "metadata": self._metadata(current),
                "updated_at": current.updated_at,
            },
            organization_id=self.organization_id,
            user_id=self.user_id,
        )
        return current

    def delete(self, entry_id: str) -> bool:
        if self.get(entry_id) is None:
            return False
        self.entries.delete_owned(
            entry_id, organization_id=self.organization_id, user_id=self.user_id
        )
        return True

    def add_upload(self, entries: list[MemoryEntry], *, suggested_name=None, dedupe=True):
        if not entries:
            return None
        from theriad.memory.store import _normalise_alias

        upload_id = str(uuid.uuid4())
        upload_name = _normalise_alias(suggested_name or "upload", fallback="upload")
        existing = {entry.text for entry in self.list()}
        added = 0
        for entry in entries:
            if not entry.text.strip() or (dedupe and entry.text in existing):
                continue
            entry.upload_id = upload_id
            entry.upload_name = upload_name
            self.entries.insert_owned(
                {
                    "id": entry.id,
                    "project_id": None,
                    "model_id": entry.model_id,
                    "label": entry.title or "Memory",
                    "content": entry.text,
                    "source": entry.source,
                    "metadata": self._metadata(entry),
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                },
                organization_id=self.organization_id,
                user_id=self.user_id,
            )
            existing.add(entry.text)
            added += 1
        return {"id": upload_id, "name": upload_name, "count": added} if added else None

    def upload_sources(self):
        groups: dict[str, dict[str, Any]] = {}
        for entry in self.list():
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
            for group in sorted(groups.values(), key=lambda item: item["updated_at"], reverse=True)
        ]

    def rename_upload(self, upload_id: str, new_name: str):
        from theriad.memory.store import _normalise_alias

        normalized = _normalise_alias(new_name)
        entries = [entry for entry in self.list() if entry.upload_id == upload_id]
        if not entries:
            raise ValueError("memory upload not found")
        for entry in entries:
            entry.upload_name = normalized
            entry.updated_at = _now()
            self.entries.update_owned(
                entry.id,
                {"metadata": self._metadata(entry), "updated_at": entry.updated_at},
                organization_id=self.organization_id,
                user_id=self.user_id,
            )
        return next(item for item in self.upload_sources() if item["id"] == upload_id)

    def get_model_profile(self, model_id: str, *, create: bool = False):
        normalized = _normalise_model_id(model_id)
        rows = self.profiles.client.list_rows(
            self.profiles.table,
            filters={
                "organization_id": f"eq.{self.organization_id}",
                "user_id": f"eq.{self.user_id}",
                "model_id": f"eq.{normalized}",
            },
            limit=1,
        )
        if not rows and create:
            now = _now()
            row = self.profiles.insert_owned(
                {
                    "model_id": normalized,
                    "context_enabled": True,
                    "learning_enabled": False,
                    "max_memories": 100,
                    "created_at": now,
                    "updated_at": now,
                },
                organization_id=self.organization_id,
                user_id=self.user_id,
            )
        elif rows:
            row = rows[0]
        else:
            return None
        count = len([entry for entry in self.list() if entry.model_id == normalized])
        return {**row, "count": count}

    def model_profiles(self):
        rows = self.profiles.list_for_owner(
            organization_id=self.organization_id, user_id=self.user_id, order="updated_at.desc"
        )
        counts: dict[str, int] = {}
        for entry in self.list():
            if entry.model_id:
                counts[entry.model_id] = counts.get(entry.model_id, 0) + 1
        return [{**row, "count": counts.get(row["model_id"], 0)} for row in rows]

    def update_model_profile(self, model_id: str, *, context_enabled=None, learning_enabled=None, max_memories=None):
        normalized = _normalise_model_id(model_id)
        current = self.get_model_profile(normalized, create=True)
        values = {
            "context_enabled": bool(current["context_enabled"] if context_enabled is None else context_enabled),
            "learning_enabled": bool(current["learning_enabled"] if learning_enabled is None else learning_enabled),
            "max_memories": max(10, min(500, int(current["max_memories"] if max_memories is None else max_memories))),
            "updated_at": _now(),
        }
        rows = self.profiles.client.update_rows(
            self.profiles.table,
            filters={
                "organization_id": f"eq.{self.organization_id}",
                "user_id": f"eq.{self.user_id}",
                "model_id": f"eq.{normalized}",
            },
            values=values,
        )
        row = rows[0] if rows else {**current, **values}
        return {**row, "count": len([entry for entry in self.list() if entry.model_id == normalized])}

    def model_memory_entries(self, model_id: str):
        normalized = _normalise_model_id(model_id)
        return [entry for entry in self.list() if entry.model_id == normalized]

    def add_model_memories(self, model_id: str, texts: list[str], *, source: str = "learned"):
        normalized = _normalise_model_id(model_id)
        profile = self.get_model_profile(normalized, create=True)
        existing = {entry.text.lower() for entry in self.model_memory_entries(normalized)}
        added = []
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
            self.entries.insert_owned(
                {
                    "id": entry.id,
                    "project_id": None,
                    "model_id": normalized,
                    "label": entry.title,
                    "content": entry.text,
                    "source": entry.source,
                    "metadata": self._metadata(entry),
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                },
                organization_id=self.organization_id,
                user_id=self.user_id,
            )
            existing.add(cleaned.lower())
            added.append(entry)
        scoped = self.model_memory_entries(normalized)
        max_memories = int(profile.get("max_memories", 100))
        for entry in sorted(scoped, key=lambda item: item.updated_at)[:-max_memories]:
            self.delete(entry.id)
        return added
