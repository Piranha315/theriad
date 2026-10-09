"""Consume real upstream streaming for paid compatibility measurements."""

from contextlib import aclosing

from .contracts import GatewayError
from .engine import bounded


async def collect_stream(transport, deployment, request):
    calls, text, usage, finish, generation = {}, "", None, None, None
    async with aclosing(bounded(transport.stream(deployment, request))) as source:
        async for chunk in source:
            generation = chunk.get("id") or generation
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text += delta.get("content") or ""
                finish = choice.get("finish_reason") or finish
                for part in delta.get("tool_calls") or []:
                    call = calls.setdefault(
                        part["index"],
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    call["id"] += part.get("id") or ""
                    for field in ("name", "arguments"):
                        call["function"][field] += part.get("function", {}).get(field) or ""
    message = {
        "role": "assistant",
        "content": text or None,
        "tool_calls": list(calls.values()) or None,
    }
    transport.validate_message(message)
    if not finish:
        raise GatewayError(
            "upstream_protocol_error", "Compatibility stream lacks a terminal choice", 409
        )
    return {
        "id": generation,
        "model": deployment.model,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": usage,
    }


async def collect_native(transport, deployment, request):
    terminal = None
    async with aclosing(bounded(transport.responses(deployment, request))) as source:
        async for event in source:
            result = event.get("response") if request.stream else event
            if result and result.get("status") in {"completed", "incomplete"}:
                terminal = result
    if terminal is None:
        raise GatewayError(
            "upstream_protocol_error", "Native compatibility response has no terminal result", 409
        )
    transport.validate_identity(terminal, deployment)
    calls, text = [], ""
    for item in terminal["output"]:
        if item["type"] == "function_call":
            calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            )
        elif item["type"] == "message":
            text += "".join(
                block["text"] for block in item["content"] if block["type"] == "output_text"
            )
    message = {"role": "assistant", "content": text or None, "tool_calls": calls or None}
    transport.validate_message(message)
    usage = terminal.get("usage") or {}
    return {
        "id": terminal["id"],
        "model": terminal["model"],
        "choices": [
            {
                "message": message,
                "finish_reason": "length"
                if terminal["status"] == "incomplete"
                else "tool_calls"
                if calls
                else "stop",
            }
        ],
        "usage": {
            **usage,
            **(
                {
                    "prompt_tokens": usage["input_tokens"],
                    "completion_tokens": usage["output_tokens"],
                }
                if "input_tokens" in usage and "output_tokens" in usage
                else {}
            ),
        },
    }
