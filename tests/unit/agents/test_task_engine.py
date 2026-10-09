from __future__ import annotations

import time
from unittest.mock import patch

from theriad.agents.task_engine import TaskEngine, ToolCall
from theriad.search.service import SearchResult


def engine_for(tmp_path):
    return TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
    )


def test_strict_write_requires_approval_then_executes(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(goal="create a file", workspace_root=str(tmp_path), permission_profile="strict")
    pending = engine.execute(task.task_id, ToolCall("project.write", {"path": "hello.txt", "content": "hello"}))
    assert pending.approval_required is True
    approval_id = task.pending_approval.approval_id
    result = engine.approve(task.task_id, approval_id, approved=True)
    assert result.ok is True
    assert (tmp_path / "hello.txt").read_text() == "hello"


def test_autonomous_workspace_write_does_not_require_approval(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(goal="create a file", workspace_root=str(tmp_path), permission_profile="autonomous")
    result = engine.execute(task.task_id, ToolCall("project.write", {"path": "hello.txt", "content": "hello"}))
    assert result.ok is True


def test_sensitive_file_read_requires_approval(tmp_path):
    (tmp_path / ".env").write_text("TOKEN=secret")
    engine = engine_for(tmp_path)
    task = engine.create(goal="inspect config", workspace_root=str(tmp_path), permission_profile="autonomous")
    result = engine.execute(task.task_id, ToolCall("project.read", {"path": ".env"}))
    assert result.approval_required is True


def test_terminal_pty_stream_and_completion(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(goal="run command", workspace_root=str(tmp_path), permission_profile="autonomous")
    started = engine.execute(task.task_id, ToolCall("terminal.start", {"command": "pwd"}))
    assert started.ok is True
    process_id = started.output["process_id"]
    deadline = time.time() + 3
    while time.time() < deadline:
        status = engine.execute(task.task_id, ToolCall("terminal.read", {"process_id": process_id}))
        if status.output["status"] != "running":
            break
        time.sleep(0.02)
    assert status.output["status"] == "completed"
    assert str(tmp_path) in status.output["output"]
    assert status.output["sandboxed"] is False


def test_events_are_append_only_and_do_not_persist_tool_output(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(goal="read", workspace_root=str(tmp_path), permission_profile="autonomous")
    (tmp_path / "note.txt").write_text("private body")
    result = engine.execute(task.task_id, ToolCall("project.read", {"path": "note.txt"}))
    assert result.ok
    events = engine.events(task.task_id)
    assert [item["sequence"] for item in events] == list(range(1, len(events) + 1))
    assert "private body" not in (tmp_path / "events" / f"{task.task_id}.jsonl").read_text()
def test_task_snapshot_restores_context_and_completion(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(
        goal="remember this result",
        workspace_root=str(tmp_path),
        permission_profile="autonomous",
        metadata={"source": "chat", "conversation_id": "conversation-1"},
    )
    completed = engine.execute(
        task.task_id,
        ToolCall("task.complete", {"answer": "durable answer"}),
    )
    assert completed.ok is True

    restored = engine_for(tmp_path).get(task.task_id)

    assert restored.status == "completed"
    assert restored.final_answer == "durable answer"
    assert restored.metadata["conversation_id"] == "conversation-1"
    assert restored.context[-1]["tool"] == "task.complete"
    assert restored.checkpoint_count > 0


def test_pending_approval_survives_restart(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(
        goal="write after approval",
        workspace_root=str(tmp_path),
        permission_profile="strict",
    )
    pending = engine.execute(
        task.task_id,
        ToolCall("project.write", {"path": "approved.txt", "content": "yes"}),
    )
    assert pending.approval_required is True
    approval_id = task.pending_approval.approval_id

    restored_engine = engine_for(tmp_path)
    restored = restored_engine.get(task.task_id)
    assert restored.status == "waiting_for_approval"
    assert restored.pending_approval.approval_id == approval_id

    result = restored_engine.approve(task.task_id, approval_id, approved=True)
    assert result.ok is True
    assert (tmp_path / "approved.txt").read_text() == "yes"


def test_running_task_recovers_to_resumable_ready_state(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(
        goal="resume me",
        workspace_root=str(tmp_path),
        permission_profile="autonomous",
    )
    task.status = "running"
    engine.checkpoint(task.task_id)

    restored = engine_for(tmp_path).get(task.task_id)

    assert restored.status == "ready"
    assert restored.events[-1].kind == "task.recovered"

def test_autonomous_task_can_search_without_browser_capability(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(
        goal="research current docs",
        workspace_root=str(tmp_path),
        permission_profile="autonomous",
        requested_capabilities=["web"],
    )

    async def search(_service, query):
        assert query == "Theriad docs"
        return [SearchResult("Docs", "https://example.org/docs", "Reference", "brave")]

    with patch("theriad.search.service.SearchService.search", new=search):
        result = engine.execute(task.task_id, ToolCall("web.search", {"query": "Theriad docs"}))

    assert result.ok is True
    assert result.output["provider"] == "brave"
    assert result.output["results"][0]["url"] == "https://example.org/docs"


def test_web_fetch_blocks_private_network_targets(tmp_path):
    engine = engine_for(tmp_path)
    task = engine.create(
        goal="read a source",
        workspace_root=str(tmp_path),
        permission_profile="autonomous",
        requested_capabilities=["web"],
    )

    result = engine.execute(
        task.task_id,
        ToolCall("web.fetch", {"url": "http://127.0.0.1:8000/private"}),
    )

    assert result.ok is False
    assert "public HTTP(S)" in (result.error or "")


def test_navigate_action_does_not_classify_url_page_text_as_risky():
    call = ToolCall(
        "browser.opensandbox.navigate",
        {"url": "data:text/html,<button>Send message</button>"},
    )

    assert TaskEngine._browser_action(call) == "navigate"
