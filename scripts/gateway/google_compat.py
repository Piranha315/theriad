"""Paid Google protocol probes through the existing shared evaluation ledger.

Only synthetic inputs are sent. This establishes compatibility, not task-quality
qualification or calibrated routing. Unknown charges retain their reservations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from fastapi.testclient import TestClient

from theriad.gateway.adoption import MonthlyBudget
from theriad.gateway.app import create_app
from theriad.gateway.contracts import Deployment
from theriad.gateway.engine import GatewayEngine
from theriad.gateway.releases import Releases
from theriad.gateway.store import LocalStore
from theriad.gateway.transport import WireTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger", default="~/.theriad/gateway/state.sqlite")
    args = parser.parse_args()
    selected = Deployment.model_validate(json.loads(Path(args.deployment).read_text()))
    if selected.provider != "vertex" or selected.region not in {"us", "eu"}:
        parser.error("an explicit verified Google multi-region deployment is required")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    store = LocalStore(output / "protocol.sqlite")
    authority = LocalStore(args.ledger)
    releases = Releases(store, secrets.token_urlsafe(48))
    bundle = {
        "deployments": [selected.model_dump()],
        "policy": {"allowed_endpoints": [selected.endpoint], "allowed_regions": [selected.region]},
        "router": None,
        "evidence": {"status": "uncalibrated Google protocol probe; no qualification"},
    }
    releases.activate(releases.install(bundle))
    engine = GatewayEngine(releases, WireTransport(), authority)
    token = secrets.token_urlsafe(32)
    principal = "eval:pilot:google-protocol"
    client = TestClient(create_app(engine, {hashlib.sha256(token.encode()).hexdigest(): principal}))
    headers = {"Authorization": "Bearer " + token}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two supplied integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool = {"type": "function", **tools[0]["function"]}
    cases = []
    report = {
        "deployment": selected.model_dump(),
        "suite": "google-protocol-v1",
        "cases": cases,
        "qualification": False,
    }

    def save():
        report["budget"] = MonthlyBudget(authority).summary()
        target = output / "report.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        os.chmod(target, 0o600)

    def request(path, body):
        response = client.post(path, json=body, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(json.dumps(response.json()))
        return response.json()

    def record(name, operation):
        start = time.monotonic()
        try:
            evidence = operation()
            cases.append(
                {
                    "case": name,
                    "passed": True,
                    "elapsed_ms": (time.monotonic() - start) * 1000,
                    **evidence,
                }
            )
        except Exception as exc:
            cases.append({"case": name, "passed": False, "error": str(exc)})
            save()
            raise
        save()

    def chat_cycle():
        messages = [
            {"role": "developer", "content": "Use the multiply tool for arithmetic."},
            {"role": "user", "content": "Use the multiply tool once to multiply 6 and 7."},
        ]
        first = request(
            "/v1/chat/completions",
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "multiply"}},
                "parallel_tool_calls": False,
                "max_tokens": 512,
                "reasoning_effort": "low",
            },
        )
        message = first["choices"][0]["message"]
        call = message["tool_calls"][0]
        arguments = json.loads(call["function"]["arguments"])
        assert arguments == {"a": 6, "b": 7}
        signature = call.get("extra_content", {}).get("google", {}).get("thought_signature")
        assert signature, "Expected a real Google tool signature"
        messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": "42"}])
        final = request(
            "/v1/chat/completions",
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": "none",
                "max_tokens": 512,
                "reasoning_effort": "low",
            },
        )
        assert "42" in final["choices"][0]["message"]["content"]
        assert final["x_theriad"]["deployment"] == first["x_theriad"]["deployment"]
        return {
            "signature_preserved": True,
            "tool_cycle_fixed": True,
            "final_usage": final["usage"],
        }

    def responses_cycle():
        prompt = {"role": "user", "content": "Use the multiply tool once to multiply 6 and 7."}
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "input": [prompt],
                "tools": [tool],
                "tool_choice": {"type": "function", "name": "multiply"},
                "parallel_tool_calls": False,
                "stream": True,
                "store": False,
                "max_output_tokens": 512,
                "reasoning": {"effort": "low", "summary": "auto"},
            },
        ) as response:
            assert response.status_code == 200
            events = [
                json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")
            ]
        assert not any(e["type"] == "response.failed" for e in events), events[-1]
        result = next(e["response"] for e in events if e["type"] == "response.completed")
        assert any(e["type"] == "response.function_call_arguments.delta" for e in events)
        assert any(
            i["type"] == "reasoning" and i.get("encrypted_content") for i in result["output"]
        )
        items = copy.deepcopy(result["output"])
        for item in items:
            item.pop("extra_content", None)
        call = next(i for i in items if i["type"] == "function_call")
        assert json.loads(call["arguments"]) == {"a": 6, "b": 7}
        final = request(
            "/v1/responses",
            {
                "input": [prompt]
                + items
                + [{"type": "function_call_output", "call_id": call["call_id"], "output": "42"}],
                "tools": [tool],
                "tool_choice": "none",
                "store": False,
                "max_output_tokens": 512,
                "reasoning": {"effort": "low", "summary": "auto"},
            },
        )
        assert "42" in "".join(
            b["text"] for i in final["output"] if i["type"] == "message" for b in i["content"]
        )
        assert store.get(f"response/{principal}/{result['id']}") is None
        assert store.get(f"response/{principal}/{final['id']}") is None
        return {
            "stream_lifecycle": True,
            "encrypted_context_replayed": True,
            "store_false": True,
            "final_usage": final["usage"],
        }

    def structured():
        result = request(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Return value equal to 42."}],
                "max_tokens": 512,
                "reasoning_effort": "low",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        assert json.loads(result["choices"][0]["message"]["content"]) == {"value": 42}
        return {"schema_validated": True, "usage": result["usage"]}

    record("chat_tools", chat_cycle)
    record("responses_stream_stateless_tools", responses_cycle)
    record("structured_output", structured)
    print(json.dumps({"passed": len(cases), "qualification": False, "budget": report["budget"]}))


if __name__ == "__main__":
    main()
