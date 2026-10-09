from __future__ import annotations

import pytest

from theriad.agents.task_engine import TaskEngine, ToolCall


def test_web_tools_are_core_and_browser_actions_remain_opt_in(tmp_path):
    import theriad.agents.task_runner as runner

    engine = TaskEngine(event_dir=tmp_path / "events")
    engine.allowed_workspace_root = tmp_path.resolve()
    task = engine.create(goal="research", workspace_root=str(tmp_path))

    tools = runner._tools_for(task)

    assert "web.search" in tools
    assert "web.fetch" in tools
    assert "browser.extension.navigate" not in tools


@pytest.mark.asyncio
async def test_runner_completes_through_typed_tool(monkeypatch, tmp_path):
    import theriad.agents.task_engine as engine_module
    import theriad.agents.task_runner as runner

    engine = TaskEngine(event_dir=tmp_path / "events")
    engine.allowed_workspace_root = tmp_path.resolve()
    task = engine.create(goal="finish", workspace_root=str(tmp_path), permission_profile="autonomous")
    monkeypatch.setattr(engine_module, "_ENGINE", engine)

    async def plan(_task, _tools):
        return ToolCall("task.complete", {"answer": "done"})

    monkeypatch.setattr(runner, "_plan", plan)
    result = await runner.run_task(task.task_id)
    assert result["status"] == "completed"
    assert result["final_answer"] == "done"


@pytest.mark.asyncio
async def test_runner_pauses_on_strict_write_approval(monkeypatch, tmp_path):
    import theriad.agents.task_engine as engine_module
    import theriad.agents.task_runner as runner

    engine = TaskEngine(event_dir=tmp_path / "events")
    engine.allowed_workspace_root = tmp_path.resolve()
    task = engine.create(goal="write", workspace_root=str(tmp_path), permission_profile="strict")
    monkeypatch.setattr(engine_module, "_ENGINE", engine)

    async def plan(_task, _tools):
        return ToolCall("project.write", {"path": "x.txt", "content": "x"})

    monkeypatch.setattr(runner, "_plan", plan)
    result = await runner.run_task(task.task_id)
    assert result["status"] == "waiting_for_approval"
    assert result["pending_approval"]["tool"] == "project.write"
    assert not (tmp_path / "x.txt").exists()
