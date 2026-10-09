import copy
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from theriad.gateway.app import create_app
from theriad.gateway.contracts import ChatRequest, EndpointPolicy, GatewayError, eligible
from theriad.gateway.engine import GatewayEngine
from theriad.gateway.provider_context import CONTEXT_KIND
from theriad.gateway.releases import Releases
from theriad.gateway.responses import ResponsesCodec
from theriad.gateway.store import LocalStore
from theriad.gateway.transport import WireTransport

from .test_vertex import deployment


@pytest.fixture
def google_gateway(tmp_path):
    selected = deployment("us").model_copy(
        update={
            "reasoning": True,
            "reasoning_config": {"effort": "low"},
            "schemas": True,
            "supported_parameters": ["parallel_tool_calls"],
        }
    )
    store = LocalStore(tmp_path / "state.sqlite")
    releases = Releases(store, "s" * 32)
    payload = {
        "deployments": [selected.model_dump()],
        "policy": {
            "allowed_endpoints": [selected.endpoint],
            "allowed_regions": ["us"],
        },
    }
    releases.activate(releases.install(payload))
    requests = []
    signature = {"google": {"thought_signature": "opaque-test-signature"}}

    class Credentials:
        async def token(self, deployment):
            return "fixture-credential"

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        calls = [
            {
                "id": f"call-{i}",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": str(i)})},
            }
            for i in range(2)
        ]
        calls[0]["extra_content"] = signature
        final = body["messages"][-1]["role"] == "tool"
        result = {
            "id": "generation",
            "model": selected.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "12" if final else None,
                        **({} if final else {"tool_calls": calls}),
                    },
                    "finish_reason": "stop" if final else "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        if not body.get("stream"):
            return httpx.Response(200, json=result)
        parts = []
        for i, call in enumerate(calls):
            head = copy.deepcopy(call)
            head.pop("extra_content", None)
            head["index"] = i
            parts.append(
                {
                    "id": "generation",
                    "model": selected.model,
                    "choices": [
                        {"index": 0, "delta": {"tool_calls": [head]}, "finish_reason": None}
                    ],
                }
            )
        # Google can provide the signature after argument chunks.
        parts.append(
            {
                "id": "generation",
                "model": selected.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "extra_content": signature}]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": result["usage"],
            }
        )
        return httpx.Response(
            200,
            content="".join("data: " + json.dumps(p) + "\n\n" for p in parts) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    engine = GatewayEngine(
        releases,
        WireTransport(transport=httpx.MockTransport(handler), credential_resolver=Credentials()),
    )
    tokens = {
        hashlib.sha256(k.encode()).hexdigest(): p
        for k, p in [("alice-key", "alice"), ("bob-key", "bob")]
    }
    return TestClient(create_app(engine, tokens)), engine, requests, payload


def headers(principal="alice"):
    return {"Authorization": "Bearer " + principal + "-key"}


def replay(result):
    # Mimic a client whose typed function items discard provider-specific fields.
    output = copy.deepcopy(result["output"])
    for item in output:
        item.pop("extra_content", None)
    return (
        [{"role": "user", "content": "Read both"}]
        + output
        + [
            {"type": "function_call_output", "call_id": f"call-{i}", "output": str(i)}
            for i in range(2)
        ]
    )


@pytest.mark.parametrize("stream", [False, True])
def test_google_stateless_parallel_context_survives_client_projection(google_gateway, stream):
    client, engine, requests, _ = google_gateway
    response = client.post(
        "/v1/responses",
        headers=headers(),
        json={
            "input": "Read both",
            "store": False,
            "stream": stream,
            "parallel_tool_calls": True,
            "reasoning": {"effort": "low", "summary": "auto"},
        },
    )
    assert response.status_code == 200
    result = (
        response.json()
        if not stream
        else next(
            json.loads(line[6:])["response"]
            for line in response.text.splitlines()
            if line.startswith("data: ")
            and json.loads(line[6:]).get("type") == "response.completed"
        )
    )
    assert engine.store.get("response/alice/" + result["id"]) is None
    assert any(i["type"] == "reasoning" and i["encrypted_content"] for i in result["output"])
    second = client.post(
        "/v1/responses",
        headers=headers(),
        json={"input": replay(result), "store": False, "parallel_tool_calls": True},
    )
    assert second.status_code == 200
    assert second.json()["output"][0]["content"][0]["text"] == "12"
    assistant = next(m for m in requests[-1]["messages"] if m.get("tool_calls"))
    assert len(assistant["tool_calls"]) == 2
    assert (
        assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
        == "opaque-test-signature"
    )
    assert requests[0]["reasoning_effort"] == "low" and "reasoning" not in requests[0]


def test_google_context_is_principal_call_and_encryption_domain_bound(google_gateway):
    client, engine, requests, _ = google_gateway
    result = client.post(
        "/v1/responses", headers=headers(), json={"input": "Read", "store": False}
    ).json()
    inputs = replay(result)
    before = len(requests)
    assert (
        client.post("/v1/responses", headers=headers("bob"), json={"input": inputs}).status_code
        == 409
    )
    changed = copy.deepcopy(inputs)
    next(i for i in changed if i.get("type") == "function_call")["arguments"] = '{"path":"changed"}'
    assert (
        client.post("/v1/responses", headers=headers(), json={"input": changed}).status_code == 409
    )
    codec = ResponsesCodec(engine.releases.key.decode(), engine.store)
    context = next(i for i in inputs if i.get("type") == "reasoning")
    # A compaction result containing attacker-controlled JSON cannot forge context.
    context["encrypted_content"] = (
        codec.cipher("alice")
        .encrypt(json.dumps({"kind": CONTEXT_KIND, "calls": {}}).encode())
        .decode()
    )
    assert (
        client.post("/v1/responses", headers=headers(), json={"input": inputs}).status_code == 409
    )
    assert len(requests) == before


def test_google_stored_continuation_and_release_freeze(google_gateway):
    client, engine, requests, payload = google_gateway
    result = client.post("/v1/responses", headers=headers(), json={"input": "Read"}).json()
    old_release = result["x_theriad"]["release"]
    updated = copy.deepcopy(payload)
    updated["deployments"][0]["id"] = "new-deployment"
    new_release = engine.releases.install(updated)
    engine.releases.activate(new_release)
    follow = client.post(
        "/v1/responses",
        headers=headers(),
        json={
            "previous_response_id": result["id"],
            "input": [
                {"type": "function_call_output", "call_id": f"call-{i}", "output": str(i)}
                for i in range(2)
            ],
        },
    )
    assert follow.status_code == 200
    assert follow.json()["x_theriad"]["release"] == old_release
    new_turn = client.post(
        "/v1/responses",
        headers=headers(),
        json={"input": replay(result) + [{"role": "user", "content": "Next task"}], "store": False},
    )
    assert new_turn.status_code == 200
    assert new_turn.json()["x_theriad"]["release"] == new_release
    assert all(
        "extra_content" not in c for m in requests[-1]["messages"] for c in m.get("tool_calls", [])
    )


@pytest.mark.parametrize("stream", [False, True])
def test_parallel_false_never_completes_a_parallel_response(google_gateway, stream):
    client, _, _, _ = google_gateway
    response = client.post(
        "/v1/responses",
        headers=headers(),
        json={"input": "Read", "parallel_tool_calls": False, "stream": stream},
    )
    if stream:
        assert "event: response.failed" in response.text
        assert "event: response.completed" not in response.text
    else:
        assert response.status_code == 502


def test_google_unsupported_reasoning_and_recursive_schema_fail_before_dispatch():
    selected = deployment("us").model_copy(update={"reasoning": True, "schemas": True})
    policy = EndpointPolicy(allowed_endpoints=[selected.endpoint], allowed_regions=["us"])
    for patch in [
        {"reasoning": {"effort": "none"}},
        {"reasoning": {"summary": "detailed"}},
        {"response_format": {"type": "json_schema", "json_schema": {"schema": {"$ref": "#"}}}},
    ]:
        req = ChatRequest(messages=[{"role": "user", "content": "Hi"}], **patch)
        assert not eligible(selected, req, policy)
        with pytest.raises(GatewayError, match="cannot preserve"):
            WireTransport.body(selected, req)


def test_google_reasoning_is_included_in_output_usage_and_budget(tmp_path):
    from theriad.gateway.contracts import billed_output_bound

    selected = deployment("us").model_copy(
        update={"reasoning": True, "reasoning_config": {"effort": "low"}}
    )
    data = {
        "usage": {
            "prompt_tokens": 44,
            "completion_tokens": 16,
            "completion_tokens_details": {"reasoning_tokens": 84},
            "total_tokens": 144,
        }
    }
    WireTransport.normalize_usage(data, selected)
    assert data["usage"]["completion_tokens"] == 100
    assert data["usage"]["provider_completion_tokens"] == 16
    WireTransport.normalize_usage(data, selected)
    assert data["usage"]["completion_tokens"] == 100  # already included, not double charged
    request = ChatRequest(messages=[{"role": "user", "content": "Hi"}], max_tokens=512)
    assert billed_output_bound(selected, request) == 1536
    data["usage"]["total_tokens"] = 200
    with pytest.raises(GatewayError, match="Ambiguous"):
        WireTransport.normalize_usage(data, selected)
