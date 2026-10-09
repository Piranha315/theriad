from fastapi.testclient import TestClient

from theriad.api.server import app


def test_models_returns_virtual_models():
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert {model["id"] for model in body["data"]} == {
        "theriad-aggressive",
        "theriad-balanced",
        "theriad-quality",
    }
    assert all(model["object"] == "model" for model in body["data"])
    assert all(model["owned_by"] == "theriad" for model in body["data"])


def test_chat_completion_maps_balanced_to_balanced_policy(monkeypatch):
    monkeypatch.setenv("THERIAD_ALLOW_OPENWEBUI_PAID_CALLS", "true")
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "theriad-balanced",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["theriad"]["policy"] == "balanced"


def test_chat_completion_env_guard_blocks_paid_calls(monkeypatch):
    monkeypatch.setenv("THERIAD_ALLOW_OPENWEBUI_PAID_CALLS", "false")
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "theriad-balanced", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 403
    assert "Paid OpenWebUI execution is disabled" in response.json()["detail"]


def test_chat_completion_non_streaming_openai_shape(monkeypatch):
    monkeypatch.setenv("THERIAD_ALLOW_OPENWEBUI_PAID_CALLS", "true")
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "theriad-quality",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Continue"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-theriad-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "theriad-quality"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert body["theriad"]["policy"] == "quality_first"
    assert body["theriad"]["fallback_chain"]


def test_streaming_returns_not_implemented_error(monkeypatch):
    monkeypatch.setenv("THERIAD_ALLOW_OPENWEBUI_PAID_CALLS", "true")
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "theriad-balanced", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )

    assert response.status_code == 400
    assert "Streaming is not implemented yet" in response.json()["detail"]
