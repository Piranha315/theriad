"""Unit tests for the memory layer, named imports, and memory injection."""
import io
import json
import zipfile

import pytest

from theriad.memory.importers import (
    ImportError_,
    import_conversation_export,
    import_uploaded_files,
)
from theriad.memory.providers import (
    build_memory_context,
    detect_mentioned_memory,
    memory_mentions,
)
from theriad.memory.store import MemoryEntry, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.json")


# ── Store ─────────────────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_add_and_list(self, store):
        entry = store.add("User prefers dark mode", tags=["ui"])
        assert store.list()[0].id == entry.id
        assert store.list()[0].tags == ["ui"]

    def test_add_empty_raises(self, store):
        with pytest.raises(ValueError):
            store.add("   ")

    def test_auto_tags_when_none_given(self, store):
        entry = store.add("User writes Python code with type hints daily")
        assert entry.tags  # keyword tags extracted

    def test_update(self, store):
        entry = store.add("original text")
        updated = store.update(entry.id, title="A useful title", text="new text", tags=["a", "b"])
        assert updated.title == "A useful title"
        assert updated.text == "new text"
        assert updated.tags == ["a", "b"]
        assert store.update("nonexistent", text="x") is None

    def test_title_round_trips_and_can_be_cleared(self, store):
        entry = store.add("remember this", title="A remembered thing")
        raw = json.loads(store.path.read_text())
        assert raw["memories"][0]["title"] == "A remembered thing"
        assert store.get(entry.id).title == "A remembered thing"

        cleared = store.update(entry.id, title="")
        assert cleared.title is None
        assert "title" not in json.loads(store.path.read_text())["memories"][0]

    def test_delete(self, store):
        entry = store.add("to delete")
        assert store.delete(entry.id) is True
        assert store.delete(entry.id) is False
        assert store.list() == []

    def test_filters(self, store):
        store.add("python things", title="Work snippets", tags=["python"], source="import")
        store.add("cooking pasta", tags=["food"], source="manual")
        assert len(store.list(source="import")) == 1
        assert len(store.list(tag="food")) == 1
        assert len(store.list(query="pasta")) == 1
        assert len(store.list(query="work snippets")) == 1
        assert len(store.list(query="python")) == 1

    def test_hand_edit_picked_up(self, store):
        store.add("first")
        # Simulate a hand edit: rewrite the file directly
        doc = json.loads(store.path.read_text())
        doc["memories"].append({"text": "hand-added memory"})
        store.path.write_text(json.dumps(doc))
        # Force mtime difference on filesystems with coarse timestamps
        import os
        os.utime(store.path, (0, 9999999999))
        texts = [e.text for e in store.list()]
        assert "hand-added memory" in texts

    def test_add_many_dedupes(self, store):
        entries = [MemoryEntry(text="same"), MemoryEntry(text="same"), MemoryEntry(text="other")]
        assert store.add_many(entries) == 2

    def test_atomic_file_shape(self, store):
        store.add("check file")
        doc = json.loads(store.path.read_text())
        assert doc["version"] == 3
        assert isinstance(doc["memories"], list)
        assert isinstance(doc["model_profiles"], dict)
        assert "_help" in doc  # hand-editors get instructions

    def test_model_memories_round_trip_with_policy(self, tmp_path):
        path = tmp_path / "model-memory.json"
        model_id = "openrouter:old/model"
        model_store = MemoryStore(path)

        profile = model_store.update_model_profile(
            model_id,
            context_enabled=True,
            learning_enabled=True,
        )
        assert profile["learning_enabled"] is True
        added = model_store.add_model_memories(model_id, ["Keep this fact"])
        assert len(added) == 1

        reloaded = MemoryStore(path)
        entries = reloaded.model_memory_entries(model_id)
        assert [entry.text for entry in entries] == ["Keep this fact"]
        assert entries[0].model_id == model_id
        saved = json.loads(path.read_text())
        assert saved["model_profiles"][model_id]["learning_enabled"] is True

    def test_import_upload_has_stable_mention_alias_and_can_be_renamed(self, store):
        upload = store.add_upload(
            [MemoryEntry(text="Use concise answers", source="import")],
            suggested_name="conversation-export",
        )

        assert upload["id"]
        assert upload["name"] == "conversation-export"
        mention = detect_mentioned_memory("Recall @conversation-export", store)
        assert mention["id"] == upload["id"]
        block, count = build_memory_context(mention, store=store)
        assert count == 1
        assert "Use concise answers" in block

        renamed = store.rename_upload(upload["id"], "preferences")
        assert renamed["name"] == "preferences"
        assert detect_mentioned_memory("Use @preferences", store)["id"] == upload["id"]
        assert [item["alias"] for item in memory_mentions(store)] == ["memory", "preferences"]


# ── Importers ─────────────────────────────────────────────────────────────────

CHATGPT_EXPORT = [
    {
        "title": "Trip planning",
        "create_time": 1730000000,
        "mapping": {
            "n1": {"message": {"author": {"role": "user"},
                               "content": {"parts": ["I live in Denver"]},
                               "create_time": 1}},
            "n2": {"message": {"author": {"role": "assistant"},
                               "content": {"parts": ["Great city!"]},
                               "create_time": 2}},
        },
    }
]

CLAUDE_EXPORT = [
    {
        "name": "Recipes",
        "created_at": "2026-01-01T00:00:00Z",
        "chat_messages": [
            {"sender": "human", "text": "I am vegetarian"},
            {"sender": "assistant", "text": "Noted."},
        ],
    }
]


class TestImporters:
    def test_mapping_export_is_not_provider_classified(self):
        fmt, entries = import_conversation_export(CHATGPT_EXPORT)
        assert fmt == "conversation"
        assert entries[0].source == "import"
        assert "chatgpt" not in entries[0].tags
        assert "Denver" in entries[0].text
        assert "Great city" not in entries[0].text  # assistant text excluded

    def test_chat_messages_export_is_not_provider_classified(self):
        fmt, entries = import_conversation_export(CLAUDE_EXPORT)
        assert fmt == "conversation"
        assert entries[0].source == "import"
        assert "vegetarian" in entries[0].text
        assert entries[0].conversation_title == "Recipes"

    def test_generic_detected(self):
        data = [{"title": "T", "provider": "gemini",
                 "messages": [{"role": "user", "content": "I like tea"}]}]
        fmt, entries = import_conversation_export(data)
        assert fmt == "generic"
        assert entries[0].source == "import"

    def test_unknown_raises(self):
        with pytest.raises(ImportError_):
            import_conversation_export([{"foo": "bar"}])

    def test_conversation_without_user_messages_skipped(self):
        data = [{"name": "Empty", "created_at": "2026-01-01T00:00:00Z",
                 "chat_messages": [{"sender": "assistant", "text": "hi"}]}]
        fmt, entries = import_conversation_export(data)
        assert fmt == "conversation"
        assert entries == []

    def test_imports_zip_and_skips_binary_members(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("claude/conversations.json", json.dumps(CLAUDE_EXPORT))
            archive.writestr("claude/image.png", b"not text")
            archive.writestr("notes.md", "# Preference\n\nI prefer short answers")
        fmt, entries, skipped = import_uploaded_files([("claude-export.zip", payload.getvalue())])
        assert fmt == "mixed"
        assert any(entry.source == "import" for entry in entries)
        assert any("short answers" in entry.text for entry in entries)
        assert skipped == 1

    def test_imports_folder_style_multiple_files_and_memory_json(self):
        fmt, entries, skipped = import_uploaded_files([
            ("Codex/memory.json", json.dumps({"memories": [{"text": "Use pytest"}]}).encode()),
            ("Codex/settings.yaml", b"theme: dark\n"),
        ])
        assert fmt == "direct"
        assert skipped == 0
        assert {entry.text for entry in entries} == {"Use pytest", "theme: dark"}


# ── Named memory injection ────────────────────────────────────────────────────

class TestMemoryContext:
    def test_build_context_respects_fixed_budget(self, store):
        for i in range(500):
            store.add(f"memory number {i} with some padding text to fill space")
        block, count = build_memory_context(store=store)
        assert len(block) <= 4000
        assert 0 < count < 500

    def test_build_context_empty_store(self, store):
        assert build_memory_context(store=store) == ("", 0)
