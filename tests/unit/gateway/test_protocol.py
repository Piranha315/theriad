import hashlib
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from theriad.gateway.app import create_app
from theriad.gateway.engine import GatewayEngine
from theriad.gateway.releases import Releases
from theriad.gateway.store import LocalStore
from theriad.gateway.transport import WireTransport


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "upstream-secret")
    store = LocalStore(tmp_path / "gateway.sqlite")
    releases = Releases(store, "s" * 32)
    release = releases.install(
        {
            "deployments": [
                {
                    "id": "test",
                    "model": "test/model",
                    "revision": "revision-1",
                    "provider": "compatible",
                    "endpoint": "https://provider.test/v1",
                    "region": "test",
                    "tiers": ["cheap_code"],
                    "context_window": 131072,
                    "max_output_tokens": 8192,
                    "tools": True,
                    "schemas": True,
                    "input_price": 1,
                    "output_price": 2,
                    "priced_at": time.time(),
                    "credential_env": "TEST_PROVIDER_KEY",
                }
            ],
            "policy": {
                "allowed_endpoints": ["https://provider.test/v1"],
                "allowed_regions": ["test"],
            },
        }
    )
    releases.activate(release)
    requests = []
    response = {
        "id": "chat-test",
        "object": "chat.completion",
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"fixture"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    def handler(request):
        requests.append(request)
        if json.loads(request.content).get("stream"):
            parts = [
                dict(
                    response,
                    choices=[
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": response["choices"][0]["message"]["tool_calls"][
                                                0
                                            ]["function"]["name"],
                                            "arguments": '{"path":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                ),
                dict(
                    response,
                    choices=[
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"fixture"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                ),
            ]
            return httpx.Response(
                200,
                content="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
                + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=response)

    engine = GatewayEngine(releases, WireTransport(transport=httpx.MockTransport(handler)))
    app = create_app(
        engine,
        {
            hashlib.sha256(b"gateway-key").hexdigest(): "alice",
            hashlib.sha256(b"bob-key").hexdigest(): "bob",
        },
    )
    return TestClient(app), engine, requests, response


def headers(token="gateway-key"):
    return {"Authorization": "Bearer " + token}


def test_preserves_tools_credentials_and_nullable_content(gateway):
    client, engine, requests, _ = gateway
    body = {
        "model": "theriad/agent",
        "messages": [
            {"role": "developer", "content": "Original instruction"},
            {"role": "user", "content": "Read"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
        "parallel_tool_calls": True,
    }
    response = client.post("/v1/chat/completions", json=body, headers=headers())
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] is None
    assert requests[0].headers["authorization"] == "Bearer upstream-secret"
    sent = json.loads(requests[0].content)
    assert sent["tools"] == body["tools"]
    assert sent["messages"] == body["messages"]
    assert sent["parallel_tool_calls"] is True
    assert engine.store.get("call/alice/call-1")["deployment"] == "test"


def test_strict_context_and_stale_price(gateway):
    client, engine, requests, _ = gateway
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x" * 150000}]},
        headers=headers(),
    )
    assert response.status_code == 422
    assert not requests
    pointer, release = engine.releases.active()
    release["deployments"][0]["priced_at"] = time.time() - 100000
    engine.releases.activate(engine.releases.install(release))
    assert (
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
            headers=headers(),
        ).status_code
        == 422
    )


def test_responses_stream_and_continuation_isolation(gateway):
    client, engine, requests, _ = gateway
    body = {
        "input": "Read",
        "stream": True,
        "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
    }
    result = client.post("/v1/responses", json=body, headers=headers())
    assert result.status_code == 200, result.text
    events = [
        json.loads(line[6:]) for line in result.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "response.completed"
    response = events[-1]["response"]
    assert response["output"][0]["arguments"] == '{"path":"fixture"}'
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    continuation = {
        "previous_response_id": response["id"],
        "input": [{"type": "function_call_output", "call_id": "call-1", "output": "file contents"}],
    }
    assert client.post("/v1/responses", json=continuation, headers=headers()).status_code == 200
    assert json.loads(requests[-1].content)["messages"][-1]["tool_call_id"] == "call-1"
    assert (
        client.post("/v1/responses", json=continuation, headers=headers("bob-key")).status_code
        == 404
    )


def test_store_false_and_unimplemented_fields(gateway):
    client, engine, _, _ = gateway
    result = client.post(
        "/v1/responses", json={"input": "Hi", "store": False}, headers=headers()
    ).json()
    assert engine.store.get("response/alice/" + result["id"]) is None
    assert (
        client.post(
            "/v1/responses",
            json={"input": "Hi", "tools": [{"type": "web_search"}]},
            headers=headers(),
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "unsupported": True},
            headers=headers(),
        ).status_code
        == 422
    )
    assert client.get("/v1/models").status_code == 401


def test_policy_revocation_overrides_tool_pin(gateway):
    client, engine, requests, _ = gateway
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers=headers(),
    )
    _, release = engine.releases.active()
    release["policy"]["allowed_endpoints"] = []
    engine.releases.activate(engine.releases.install(release))
    result = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "Read"},
            ]
        },
        headers=headers(),
    )
    assert result.status_code == 422
    assert len(requests) == 1


def test_nullable_input_and_historical_tool_cycles(gateway):
    client, engine, requests, _ = gateway
    engine.store.put("call/alice/old", {"release": "historical", "deployment": "old"})
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "old",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old", "content": "old data"},
            {"role": "user", "content": "A fresh turn"},
        ]
    }
    assert client.post("/v1/chat/completions", json=body, headers=headers()).status_code == 200
    assert json.loads(requests[0].content)["messages"][0]["content"] is None


def test_native_namespace_and_store_false(gateway):
    client, engine, requests, _ = gateway
    _, bundle = engine.releases.active()
    bundle["deployments"][0]["native_responses"] = True
    engine.releases.activate(engine.releases.install(bundle))

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["store"] is False
        assert "client_metadata" not in body
        name = body["tools"][0]["name"]
        return httpx.Response(
            200,
            json={
                "id": "resp-native",
                "object": "response",
                "status": "completed",
                "model": "test/model",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "native-call",
                        "name": name,
                        "arguments": "{}",
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost": 0.001},
            },
        )

    engine.transport.transport = httpx.MockTransport(handler)
    body = {
        "input": "Read",
        "store": False,
        "client_metadata": {"session_id": "fixture"},
        "tools": [
            {
                "type": "namespace",
                "name": "files",
                "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}],
            }
        ],
    }
    response = client.post("/v1/responses", json=body, headers=headers())
    assert response.status_code == 200, response.text
    assert response.json()["output"][0]["namespace"] == "files"
    assert response.json()["output"][0]["name"] == "read"
    assert engine.store.get("native/alice/resp-native") is None
    assert engine.store.get("call/alice/native-call")


def test_native_opaque_reasoning_binding_and_continuation(gateway):
    client, engine, _, _ = gateway
    _, bundle = engine.releases.active()
    bundle["deployments"][0].update(native_responses=True, reasoning=True)
    bound = engine.releases.install(bundle)
    engine.releases.activate(bound)
    response = {
        "id": "resp-bound",
        "object": "response",
        "status": "completed",
        "model": "test/model",
        "output": [
            {"type": "reasoning", "id": "rs-1", "encrypted_content": "opaque-value", "summary": []},
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done", "annotations": []}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    engine.transport.transport = httpx.MockTransport(handler)
    assert client.post("/v1/responses", json={"input": "Hi"}, headers=headers()).status_code == 200
    bundle["deployments"][0]["id"] = "new"
    engine.releases.activate(engine.releases.install(bundle))
    followup = client.post(
        "/v1/responses",
        json={"previous_response_id": "resp-bound", "input": "Continue"},
        headers=headers(),
    )
    assert followup.status_code == 200, followup.text
    assert followup.json()["x_theriad"]["release"] == bound
    assert sent[-1]["input"][1]["encrypted_content"] == "opaque-value"
    assert (
        client.post(
            "/v1/responses",
            json={"input": [{"type": "reasoning", "encrypted_content": "unrecognized"}]},
            headers=headers(),
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/responses",
            json={"previous_response_id": "resp-bound", "input": "Hi"},
            headers=headers("bob-key"),
        ).status_code
        == 404
    )


def test_stream_failure_never_restarts_after_output(gateway):
    client, engine, requests, _ = gateway
    _, bundle = engine.releases.active()
    bundle["deployments"].append({**bundle["deployments"][0], "id": "fallback", "input_price": 2})
    engine.releases.activate(engine.releases.install(bundle))

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            content='data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    engine.transport.transport = httpx.MockTransport(handler)
    result = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
        headers=headers(),
    )
    assert "partial" in result.text
    assert "upstream_protocol_error" in result.text
    assert "[DONE]" not in result.text
    assert len(requests) == 1


def test_fallback_credential_isolation_and_policy_recheck(gateway, monkeypatch):
    client, engine, requests, response = gateway
    monkeypatch.setenv("FALLBACK_KEY", "fallback-secret")
    _, bundle = engine.releases.active()
    bundle["deployments"].append(
        {
            **bundle["deployments"][0],
            "id": "fallback",
            "input_price": 2,
            "endpoint": "https://fallback.test/v1",
            "credential_env": "FALLBACK_KEY",
        }
    )
    bundle["policy"]["allowed_endpoints"].append("https://fallback.test/v1")
    engine.releases.activate(engine.releases.install(bundle))

    def handler(request):
        requests.append(request)
        if request.url.host == "provider.test":
            return httpx.Response(503)
        assert request.headers["authorization"] == "Bearer fallback-secret"
        return httpx.Response(200, json=response)

    engine.transport.transport = httpx.MockTransport(handler)
    body = {"messages": [{"role": "user", "content": "Hi"}]}
    assert client.post("/v1/chat/completions", json=body, headers=headers()).status_code == 200
    assert len(requests) == 2
    requests.clear()

    def revoke(request):
        requests.append(request)
        _, current = engine.releases.active()
        current["policy"]["allowed_endpoints"] = ["https://provider.test/v1"]
        engine.releases.activate(engine.releases.install(current))
        return httpx.Response(503)

    engine.transport.transport = httpx.MockTransport(revoke)
    assert client.post("/v1/chat/completions", json=body, headers=headers()).status_code == 422
    assert len(requests) == 1


def test_compaction_is_explicit_and_principal_scoped(gateway):
    client, engine, requests, response = gateway

    def handler(request):
        requests.append(request)
        output = {
            **response,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Preserved task summary"},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(200, json=output)

    engine.transport.transport = httpx.MockTransport(handler)
    compact = client.post("/v1/responses/compact", json={"input": "Task"}, headers=headers())
    assert compact.status_code == 200, compact.text
    item = compact.json()["output"][0]
    assert "Preserved task summary" not in item["encrypted_content"]
    assert (
        client.post(
            "/v1/responses",
            json={"input": [item, {"role": "user", "content": "Continue"}]},
            headers=headers(),
        ).status_code
        == 200
    )
    assert (
        client.post("/v1/responses", json={"input": [item]}, headers=headers("bob-key")).status_code
        == 400
    )


def test_outcome_reporting_is_scoped_and_idempotent(gateway):
    client, engine, _, _ = gateway
    reply = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers=headers(),
    ).json()
    episode = reply["x_theriad"]["episode_id"]
    assert (
        client.post(
            f"/v1/theriad/episodes/{episode}/outcome", json={"success": True}, headers=headers()
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/v1/theriad/episodes/{episode}/outcome",
            json={"success": True},
            headers=headers("bob-key"),
        ).status_code
        == 404
    )
    assert engine.store.get(f"episode/alice/{episode}")["outcome"]["success"] is True


def test_unknown_outstanding_tool_binding_fails_without_rerouting(gateway):
    client, _, requests, _ = gateway
    result = client.post(
        "/v1/chat/completions",
        headers=headers(),
        json={"messages": [{"role": "tool", "tool_call_id": "not-ours", "content": "result"}]},
    )
    assert result.status_code == 409
    assert result.json()["error"]["code"] == "tool_cycle_unavailable"
    assert not requests


def test_loopback_byok_is_explicit_and_never_becomes_other_provider_credential(
    gateway, monkeypatch
):
    client, engine, requests, _ = gateway
    _, payload = engine.releases.active()
    payload["deployments"][0].update(
        provider="openrouter",
        endpoint="https://openrouter.ai/api/v1",
        hosting_provider="approved-host",
        serving_variant="fp16",
    )
    payload["policy"]["allowed_endpoints"] = ["https://openrouter.ai/api/v1"]
    engine.releases.activate(engine.releases.install(payload))
    local = TestClient(client.app, client=("127.0.0.1", 40000))
    body = {"messages": [{"role": "user", "content": "hello"}]}
    assert (
        local.post(
            "/v1/chat/completions", headers=headers("legacy-provider-key"), json=body
        ).status_code
        == 401
    )
    monkeypatch.setenv("THERIAD_GATEWAY_LOOPBACK_BYOK", "true")
    assert (
        local.post(
            "/v1/chat/completions", headers=headers("legacy-provider-key"), json=body
        ).status_code
        == 200
    )
    assert requests[-1].headers["authorization"] == "Bearer legacy-provider-key"
    remote = TestClient(client.app, client=("198.51.100.1", 40000))
    assert (
        remote.post(
            "/v1/chat/completions", headers=headers("legacy-provider-key"), json=body
        ).status_code
        == 401
    )
    # The ordinary scoped gateway key still resolves deployment credentials.
    assert local.post("/v1/chat/completions", headers=headers(), json=body).status_code == 200
    assert requests[-1].headers["authorization"] == "Bearer upstream-secret"


@pytest.mark.parametrize("stream", [False, True])
def test_translated_namespace_calls_preserve_identity_and_continuation(gateway, stream):
    client, engine, requests, response = gateway
    alias = "ns_" + hashlib.sha256(b"files/read_file").hexdigest()[:40]
    response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = alias
    body = {
        "model": "theriad/agent",
        "input": "Read the fixture",
        "stream": stream,
        "tools": [
            {
                "type": "namespace",
                "name": "files",
                "tools": [
                    {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
                ],
            }
        ],
    }
    result = client.post("/v1/responses", headers=headers(), json=body)
    assert result.status_code == 200
    if stream:
        events = [
            json.loads(line[6:]) for line in result.text.splitlines() if line.startswith("data: ")
        ]
        output = next(e["response"] for e in events if e["type"] == "response.completed")
        item = next(e["item"] for e in events if e["type"] == "response.output_item.done")
        assert item["namespace"] == "files" and item["name"] == "read_file"
    else:
        output = result.json()
    assert output["output"][0]["namespace"] == "files"
    assert output["output"][0]["name"] == "read_file"
    continuation = client.post(
        "/v1/responses",
        headers=headers(),
        json={
            **body,
            "stream": False,
            "previous_response_id": output["id"],
            "input": [
                {"type": "function_call_output", "call_id": "call-1", "output": "fixture contents"}
            ],
        },
    )
    assert continuation.status_code == 200
    wire = json.loads(requests[-1].content)
    assert wire["tools"][0]["function"]["name"] == alias
    assert any(
        m.get("tool_calls", [{}])[0].get("function", {}).get("name") == alias
        for m in wire["messages"]
        if m.get("tool_calls")
    )


def test_catalog_parallel_constraint_fails_before_dispatch(gateway):
    client, engine, requests, _ = gateway
    payload = engine.releases.active()[1]
    payload["deployments"][0]["supported_parameters"] = ["tools", "tool_choice", "max_tokens"]
    engine.releases.activate(engine.releases.install(payload))
    result = client.post(
        "/v1/chat/completions",
        headers=headers(),
        json={
            "model": "theriad/agent",
            "messages": [{"role": "user", "content": "Read"}],
            "parallel_tool_calls": False,
        },
    )
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "no_eligible_model"
    assert requests == []
