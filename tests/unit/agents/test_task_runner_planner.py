from __future__ import annotations

import pytest

from theriad.agents.task_engine import TaskEngineError
from theriad.agents.task_runner import _parse_planner_decision


TOOLS = {
    "task.complete": {},
    "browser.opensandbox.observe": {},
}


def test_planner_accepts_json_wrapped_in_short_prose():
    decision = _parse_planner_decision(
        'I will inspect the page first.\n```json\n'
        '{"tool":"browser.opensandbox.observe","arguments":{"screenshot":false}}'
        '\n```',
        TOOLS,
    )

    assert decision.name == "browser.opensandbox.observe"
    assert decision.arguments == {"screenshot": False}


def test_planner_rejects_truncated_json():
    with pytest.raises(TaskEngineError, match="invalid JSON"):
        _parse_planner_decision('{"tool":"task.complete","arguments":', TOOLS)


def test_planner_rejects_unavailable_tool():
    with pytest.raises(TaskEngineError, match="unavailable tool"):
        _parse_planner_decision(
            '{"tool":"browser.opensandbox.click","arguments":{}}',
            TOOLS,
        )
