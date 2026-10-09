from fastapi.testclient import TestClient

from theriad.api.server import app
from theriad.execution import ExecutionEvidence


def test_execute_search_like_task_has_non_empty_final_answer():
    response = TestClient(app).post(
        "/api/execute",
        json={"prompt": "Find the capital of France."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["final_answer"].strip()
    assert body["content"] == body["final_answer"]
    assert body["trace"]


def test_execute_synthesis_receives_browser_observations(monkeypatch):
    captured = {}

    async def mocked_execute_phase(prompt, route):
        return ExecutionEvidence(
            trace=[
                {
                    "type": "page_visit",
                    "title": "Example Support",
                    "url": "https://example.com/support",
                }
            ],
            artifacts=[
                {
                    "type": "observation",
                    "content": "Support email: help@example.com",
                    "source_url": "https://example.com/support",
                }
            ],
        )

    async def mocked_synthesis(prompt, evidence, route):
        captured["artifacts"] = evidence.artifacts
        return (
            "Contact help@example.com. Source: Example Support "
            "(https://example.com/support)"
        )

    monkeypatch.setattr("theriad.execution.execute_phase", mocked_execute_phase)
    monkeypatch.setattr("theriad.execution.synthesize_final_answer", mocked_synthesis)

    body = TestClient(app).post(
        "/api/execute",
        json={"prompt": "How do I contact support?"},
    ).json()

    assert "help@example.com" in captured["artifacts"][0]["content"]
    assert "help@example.com" in body["final_answer"]
    assert "https://example.com/support" in body["final_answer"]


def test_execute_failure_has_user_facing_final_answer(monkeypatch):
    async def failing_execute_phase(prompt, route):
        raise RuntimeError("browser timed out")

    monkeypatch.setattr("theriad.execution.execute_phase", failing_execute_phase)

    body = TestClient(app).post(
        "/api/execute",
        json={"prompt": "Open the unavailable page."},
    ).json()

    assert body["status"] == "failed"
    assert "couldn't complete" in body["final_answer"]
    assert "browser timed out" in body["final_answer"]


def test_empty_completed_synthesis_is_guarded(monkeypatch):
    async def empty_synthesis(prompt, evidence, route):
        return "   "

    monkeypatch.setattr("theriad.execution.synthesize_final_answer", empty_synthesis)

    body = TestClient(app).post(
        "/api/execute",
        json={"prompt": "Do something."},
    ).json()

    assert body["status"] == "failed"
    assert body["final_answer"] == "Execution completed, but no final answer was generated."
