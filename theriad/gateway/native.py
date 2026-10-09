"""Lossless native Responses requests; unsupported hosted tools fail closed."""

from __future__ import annotations

import copy
import hashlib
import json

from .contracts import ChatRequest, GatewayError
from .responses import ResponsesCodec


class NativeCodec(ResponsesCodec):
    def prepare(self, body, principal):
        if not isinstance(body, dict):
            raise GatewayError("invalid_request", "Responses body must be an object", 422)
        inputs = body.get("input") or []
        if isinstance(inputs, list) and any(
            item.get("type") == "reasoning" and self.provider_context(item, principal) is not None
            for item in inputs
        ):
            # Gateway-encrypted Google tool state is translated locally, never sent
            # as if it were native OpenAI reasoning.
            return self.parse(body, principal)[0]
        supported = {
            "model",
            "input",
            "instructions",
            "previous_response_id",
            "stream",
            "store",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_output_tokens",
            "temperature",
            "top_p",
            "text",
            "reasoning",
            "metadata",
            "include",
            "truncation",
            "prompt_cache_key",
            "service_tier",
            "background",
            "x_theriad_session",
            "client_metadata",
        }
        if set(body) - supported:
            raise GatewayError(
                "unsupported_field", f"Unsupported fields: {sorted(set(body) - supported)}"
            )
        if body.get("background") or body.get("truncation", "disabled") != "disabled":
            raise GatewayError(
                "unsupported_field", "Background and implicit truncation are unavailable"
            )
        tools, names = [], {}
        for tool in body.get("tools") or []:
            if tool.get("type") == "namespace":
                namespace = tool.get("name")
                if not isinstance(namespace, str) or not namespace:
                    raise GatewayError("invalid_tool", "Namespace name is required", 422)
                for child in tool.get("tools", []):
                    if child.get("type") not in {"function", "custom"}:
                        raise GatewayError(
                            "unsupported_tool", "Namespace children must be caller-executed tools"
                        )
                    if not isinstance(child.get("name"), str) or not child["name"]:
                        raise GatewayError("invalid_tool", "Tool name is required", 422)
                    wire_name = (
                        "ns_"
                        + hashlib.sha256((namespace + "/" + child["name"]).encode()).hexdigest()[
                            :40
                        ]
                    )
                    names[wire_name] = (namespace, child["name"])
                    tools.append(
                        {
                            **child,
                            "name": wire_name,
                            "description": namespace + ": " + child.get("description", ""),
                        }
                    )
            else:
                tools.append(tool)
        if any(
            not isinstance(t, dict) or t.get("type") not in {"function", "custom"} for t in tools
        ):
            raise GatewayError(
                "unsupported_tool", "Only caller-executed functions/custom tools are supported"
            )
        if any(not isinstance(t.get("name"), str) or not t["name"] for t in tools):
            raise GatewayError("invalid_tool", "Tool name is required", 422)
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        if not isinstance(inputs, list) or any(not isinstance(i, dict) for i in inputs):
            raise GatewayError("invalid_request", "input must be text or an item array", 422)
        previous = body.get("previous_response_id")
        if previous:
            history = self.store.get(f"native/{principal}/{previous}")
            if not history:
                if not self.store.get(f"response/{principal}/{previous}"):
                    raise GatewayError(
                        "response_not_found",
                        "Previous response unavailable for this principal",
                        404,
                    )
                # The converted adapter owns converted histories. Return a request
                # that can only be served by that adapter; don't cross histories.
                request, _ = self.parse(body, principal)
                return request
            inputs = history["input"] + history["output"] + inputs
        inputs = copy.deepcopy(inputs)
        for index, item in enumerate(inputs):
            if item.get("type") in {
                "function_call",
                "custom_tool_call",
                "function_call_output",
                "custom_tool_call_output",
            } and (not isinstance(item.get("call_id"), str) or not item["call_id"]):
                raise GatewayError("invalid_input", "Tool items require call_id", 422)
            if item.get("type") in {"function_call", "custom_tool_call"} and not isinstance(
                item.get("name"), str
            ):
                raise GatewayError("invalid_input", "Tool calls require a name", 422)
            if item.get("type") == "compaction":
                from cryptography.fernet import InvalidToken

                try:
                    text = (
                        self.cipher(principal).decrypt(item["encrypted_content"].encode()).decode()
                    )
                except (InvalidToken, KeyError) as exc:
                    raise GatewayError("invalid_compaction", "Compaction cannot be read") from exc
                inputs[index] = {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            elif item.get("type", "message") not in {
                "message",
                "function_call",
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "reasoning",
            }:
                raise GatewayError("unsupported_input", "Unsupported native input item")
            elif item.get("type", "message") == "message" and isinstance(item.get("content"), list):
                if any(
                    not isinstance(b, dict)
                    or b.get("type") not in {"input_text", "output_text"}
                    or not isinstance(b.get("text"), str)
                    for b in item["content"]
                ):
                    raise GatewayError(
                        "unsupported_content", "Native pilot accepts text content only"
                    )
        wire = {
            k: v
            for k, v in body.items()
            if k
            not in {
                "previous_response_id",
                "metadata",
                "x_theriad_session",
                "store",
                "client_metadata",
            }
        }
        wire["input"] = inputs
        wire["tools"] = tools
        choice = wire.get("tool_choice")
        if isinstance(choice, dict) and choice.get("namespace"):
            replacement = next(
                (
                    name
                    for name, pair in names.items()
                    if pair == (choice["namespace"], choice.get("name"))
                ),
                None,
            )
            if not replacement:
                raise GatewayError(
                    "tool_namespace_missing", "Chosen tool namespace is unavailable", 422
                )
            wire["tool_choice"] = {k: v for k, v in choice.items() if k != "namespace"} | {
                "name": replacement
            }
        for item in inputs:
            if item.get("namespace") and item.get("name"):
                wire_name = next(
                    (
                        name
                        for name, pair in names.items()
                        if pair == (item["namespace"], item["name"])
                    ),
                    None,
                )
                if not wire_name:
                    raise GatewayError(
                        "tool_namespace_missing", "Historical namespace tool absent from request"
                    )
                item["name"] = wire_name
                item.pop("namespace")
        # Routing projection includes every byte of opaque items, schemas and settings.
        # This projection never goes to the upstream deployment.
        messages = [{"role": "user", "content": json.dumps(wire, ensure_ascii=False)}]
        for item in reversed(inputs):
            if item.get("type") in {"function_call_output", "custom_tool_call_output"}:
                messages.append({"role": "tool", "content": "", "tool_call_id": item["call_id"]})
            else:
                break
        request = ChatRequest(
            model=body.get("model", "theriad/agent"),
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {"name": t.get("name", ""), "parameters": t.get("parameters", {})},
                }
                for t in tools
            ]
            or None,
            stream=body.get("stream", False),
            max_completion_tokens=body.get("max_output_tokens"),
            reasoning=body.get("reasoning"),
            response_format=(body.get("text") or {}).get("format"),
            x_theriad_session=body.get("x_theriad_session"),
        )
        request._native_body = wire
        request._native_custom = any(t["type"] == "custom" for t in tools)
        request._namespace_map = names
        pins = []
        for item in inputs:
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                pin = self.store.get(
                    "opaque/"
                    + principal
                    + "/"
                    + hashlib.sha256(item["encrypted_content"].encode()).hexdigest()
                )
                if not pin:
                    raise GatewayError(
                        "opaque_reasoning_unsupported",
                        "Opaque reasoning has no verified deployment binding",
                    )
                pins.append(pin)
        if pins:
            if any(pin != pins[0] for pin in pins):
                raise GatewayError(
                    "opaque_reasoning_conflict",
                    "Opaque items belong to different deployment releases",
                )
            request._deployment_pin = pins[0]
        return request

    def save_native(self, principal, request, result):
        self.store.put(
            f"native/{principal}/{result['id']}",
            {"input": request._native_body["input"], "output": result["output"]},
            86400,
        )

    @staticmethod
    def restore_names(request, event):
        items = event.get("output", []) + (event.get("response") or {}).get("output", [])
        if event.get("item"):
            items = [*items, event["item"]]
        for item in items:
            if item.get("name") in request._namespace_map:
                item["namespace"], item["name"] = request._namespace_map[item["name"]]
