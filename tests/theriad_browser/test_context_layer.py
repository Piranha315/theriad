import json

from fastapi.testclient import TestClient

from theriad_browser.chat import handle_chat_message
from theriad.api.server import app
from theriad.context_layer import (
    ImportInput,
    MemorySource,
    SecondBrainService,
    no_results_answer,
    parse_context_command,
)


def test_memory_card_crud(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")

    saved = service.save(
        kind="project_decision",
        project="theriad_browser",
        title="Use explicit recall",
        summary="Theriad should recall context only when @source commands require it.",
        tags=["routing"],
        source=MemorySource(type="manual", provider="chat"),
    )

    assert service.read([saved.id])[0].title == "Use explicit recall"
    assert service.search(query="explicit recall", projects=["theriad_browser"])[0].id == saved.id
    assert service.forget(saved.id) is True
    assert service.search(query="explicit recall", projects=["theriad_browser"]) == []


def test_search_filters_by_provider(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")
    service.save(
        kind="conversation_summary",
        title="Claude model routing note",
        summary="Claude recommended provider-aware memory lookup.",
        source={"type": "conversation_import", "provider": "claude"},
    )
    service.save(
        kind="conversation_summary",
        title="ChatGPT browser note",
        summary="ChatGPT discussed browser bridges.",
        source={"type": "conversation_import", "provider": "chatgpt"},
    )

    results = service.search(query="note", sourceProviders=["claude"])

    assert len(results) == 1
    assert results[0].source.provider == "claude"


def test_search_filters_by_project(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")
    service.save(
        kind="technical_note",
        project="theriad_browser",
        title="START_URL",
        summary="START_URL controls the Theriad local browser entry point.",
    )
    service.save(
        kind="technical_note",
        project="kalshi",
        title="Kalshi priors",
        summary="Kalshi priors are loaded separately.",
    )

    results = service.search(query="START_URL", projects=["theriad_browser"])

    assert [result.project for result in results] == ["theriad_browser"]


def test_generic_transcript_import_creates_memory(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")

    result = service.import_context(
        ImportInput(
            provider="generic",
            title="Routing transcript",
            text="User:\nWhat about model routing?\n\nAssistant:\nUse explicit context recall for Theriad.",
        )
    )

    assert result["importedConversations"] == 1
    assert result["importedMessages"] == 2
    assert result["memoryCardsCreated"] == 1
    found = service.search(query="explicit context recall", projects=["theriad_browser"])
    assert found[0].source.provider == "generic"


def test_claude_style_transcript_import(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")

    result = service.import_context(
        {
            "provider": "claude",
            "sourceFileName": "claude.md",
            "text": "Human:\nShould we stuff all memory into prompts?\n\nClaude:\nNo, use explicit searchable memory cards.",
        }
    )

    assert result["provider"] == "claude"
    assert service.search(query="searchable memory cards", sourceProviders=["claude"])


def test_chatgpt_json_import(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")
    payload = {
        "title": "Open Router Context",
        "messages": [
            {"id": "m1", "role": "user", "content": "What did we decide about OpenRouter?"},
            {"id": "m2", "role": "assistant", "content": "Keep context recall explicit and provider-aware."},
        ],
    }

    result = service.import_context({"provider": "chatgpt", "jsonData": payload})

    assert result["provider"] == "chatgpt"
    assert service.search(query="provider-aware", sourceProviders=["chatgpt"], projects=["theriad_browser"])


def test_obsidian_markdown_import(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")

    result = service.import_context(
        {
            "provider": "obsidian",
            "sourceFileName": "Chrome MCP.md",
            "text": "# Chrome MCP\n\nTheriad notes say Chrome MCP should be source-aware.",
        }
    )

    assert result["provider"] == "obsidian"
    assert service.search(query="Chrome MCP", sourceProviders=["obsidian"], projects=["theriad_browser"])


def test_source_parsing_with_recent_context():
    parsed = parse_context_command(
        "What did @claude think of this idea?",
        recent_context="Explicit @provider commands for context layer retrieval.",
    )

    assert parsed.action == "recall"
    assert parsed.sources == [{"kind": "provider", "id": "claude"}]
    assert parsed.requiresRetrieval is True
    assert "@claude" not in parsed.query
    assert parsed.rewrittenQuery.startswith("Explicit @provider commands")


def test_remember_parsing():
    parsed = parse_context_command("/remember this as an Theriad design decision")

    assert parsed.action == "remember"
    assert parsed.requiresRetrieval is False
    assert parsed.query == "this as an Theriad design decision"


def test_multiple_sources_compare():
    parsed = parse_context_command("Compare @claude and @chatgpt on this", "Model routing decisions")

    assert parsed.action == "recall"
    assert parsed.requiresRetrieval is True
    assert {"kind": "provider", "id": "claude"} in parsed.sources
    assert {"kind": "provider", "id": "chatgpt"} in parsed.sources
    assert "Model routing decisions" in parsed.rewrittenQuery


def test_no_results_behavior_does_not_hallucinate_provider_opinion(tmp_path):
    service = SecondBrainService(tmp_path / "context.sqlite")

    parsed, results = service.recall_for_message("What did @claude think about START_URL?")

    assert results == []
    assert no_results_answer(parsed) == "I don't have imported claude context on this yet."


def test_chat_source_recall_does_not_trigger_live_claude(monkeypatch, tmp_path):
    db_path = tmp_path / "context.sqlite"
    service = SecondBrainService(db_path)
    service.import_context(
        {
            "provider": "claude",
            "title": "Theriad START_URL",
            "text": "Human:\nWhat is START_URL?\nClaude:\nSTART_URL is the local browser launch target.",
        }
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("theriad_browser.chat.SecondBrainService", lambda: SecondBrainService(db_path))

    response = handle_chat_message("What did @claude say about START_URL?")

    assert response.intent.value == "normal_chat"
    assert response.agent_run_id is None
    assert "local browser launch target" in response.content
    assert response.trace[0]["type"] == "second_brain.search"


def test_context_api_endpoints(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    imported = client.post(
        "/api/context/import",
        json={
            "provider": "claude",
            "title": "Browser bridge",
            "text": "Human:\nChrome MCP?\nClaude:\nUse a browser bridge with provenance.",
        },
    )
    assert imported.status_code == 200
    assert imported.json()["memoryCardsCreated"] == 1

    searched = client.post("/api/context/search", json={"query": "browser bridge", "sourceProviders": ["claude"]})
    assert searched.status_code == 200
    assert searched.json()["results"][0]["source"]["provider"] == "claude"

    parsed = client.post("/api/context/parse-command", json={"message": "@claude browser bridge"})
    assert parsed.status_code == 200
    assert parsed.json()["results"]


def test_context_panel_contract():
    source = open("frontend/src/ContextPanel.jsx", encoding="utf-8").read()

    assert "data-testid=\"context-panel\"" in source
    assert "data-testid=\"context-import-form\"" in source
    assert "data-testid=\"context-save-form\"" in source
    assert "data-testid=\"context-search-form\"" in source
    assert "data-testid=\"context-memory-browser\"" in source
    assert "data-testid=\"context-results\"" in source
    assert "@claude" in source
    assert "Upload text/Markdown transcript" in source
    assert "Remember Manually" in source
    assert "Stored Memories" in source
    assert "/api/context/import" in source
    assert "/api/context/save" in source
    assert "/api/context/forget" in source
    assert "/api/context/memories" in source
    assert "/api/context/parse-command" in source
