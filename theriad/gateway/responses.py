"""Responses wire adapter over real streamed upstream inference."""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid

from cryptography.fernet import Fernet, InvalidToken

from .contracts import ChatRequest, GatewayError
from .provider_context import (
    CONTEXT_ID_PREFIX,
    CONTEXT_KIND,
    bind_calls,
    merge_extra_content,
    restore_calls,
)


def namespace_projection(body):
    """Flatten caller function namespaces without changing their schemas or identities."""
    import copy

    body = copy.deepcopy(body)
    tools, names = [], {}
    for tool in body.get("tools") or []:
        if tool.get("type") != "namespace":
            tools.append(tool)
            continue
        namespace = tool.get("name")
        if not isinstance(namespace, str) or not namespace:
            raise GatewayError("invalid_tool", "Namespace name is required", 422)
        for child in tool.get("tools", []):
            name = child.get("name")
            if child.get("type") != "function" or not isinstance(name, str) or not name:
                raise GatewayError(
                    "unsupported_tool", "Translated namespaces require functions", 422
                )
            alias = "ns_" + hashlib.sha256((namespace + "/" + name).encode()).hexdigest()[:40]
            names[alias] = (namespace, name)
            tools.append(
                {
                    **child,
                    "name": alias,
                    "description": namespace + ": " + child.get("description", ""),
                }
            )
    body["tools"] = tools
    items = body.get("input", [])
    choice = body.get("tool_choice")
    for item in (items if isinstance(items, list) else []) + (
        [choice] if isinstance(choice, dict) else []
    ):
        if item.get("namespace") and item.get("name"):
            alias = next(
                (key for key, pair in names.items() if pair == (item["namespace"], item["name"])),
                None,
            )
            if not alias:
                raise GatewayError("tool_namespace_missing", "Namespace tool is unavailable", 422)
            item["name"] = alias
            item.pop("namespace")
    return body, names


class ResponsesCodec:
    def __init__(self, secret: str, store):
        self.secret, self.store = secret, store

    def cipher(self, principal):
        return Fernet(
            base64.urlsafe_b64encode(hashlib.sha256((self.secret + principal).encode()).digest())
        )

    def provider_context(self, item, principal):
        token = item.get("encrypted_content")
        if not token:
            return None
        try:
            value = json.loads(self.provider_cipher(principal).decrypt(token.encode(), ttl=86400))
            if value.get("kind") != CONTEXT_KIND:
                return None
            if not isinstance(value.get("calls"), dict):
                raise ValueError("invalid context")
            return value["calls"]
        except (InvalidToken, ValueError, TypeError, AttributeError):
            if item.get("id", "").startswith(CONTEXT_ID_PREFIX):
                raise GatewayError(
                    "invalid_provider_context",
                    "Provider context is invalid, expired or foreign",
                    409,
                ) from None
            return None

    def provider_cipher(self, principal):
        # Separate from compaction encryption: model-generated summaries must
        # never become trusted provider-state envelopes.
        return Fernet(
            base64.urlsafe_b64encode(
                hashlib.sha256(
                    ("provider-context-v1\0" + self.secret + "\0" + principal).encode()
                ).digest()
            )
        )

    def parse(self, body, principal):
        body, namespaces = namespace_projection(body)
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
            "prompt_cache_retention",
            "service_tier",
            "background",
            "user",
            "x_theriad_session",
            "context_management",
            "client_metadata",
        }
        unknown = set(body) - supported
        if unknown:
            raise GatewayError(
                "unsupported_field", f"Unsupported Responses fields: {sorted(unknown)}"
            )
        if (
            body.get("background")
            or body.get("truncation", "disabled") != "disabled"
            or body.get("context_management")
        ):
            raise GatewayError(
                "unsupported_field",
                "Background generation, server truncation and automatic compaction are not supported",
            )
        if body.get("service_tier", "auto") not in {"auto", "default"}:
            raise GatewayError("unsupported_field", "Requested service tier is unavailable")
        if body.get("prompt_cache_retention"):
            raise GatewayError("unsupported_field", "Explicit cache retention is unavailable")
        include = body.get("include") or []
        if any(i != "reasoning.encrypted_content" for i in include):
            raise GatewayError("unsupported_field", "Requested include field is unavailable")
        messages = []
        provider_bindings = {}
        previous = body.get("previous_response_id")
        if previous:
            record = self.store.get(f"response/{principal}/{previous}")
            if not record:
                raise GatewayError(
                    "response_not_found",
                    "Previous response is missing, expired or belongs to another principal",
                    404,
                )
            messages.extend(record["messages"])
            provider_bindings.update(record.get("provider_bindings", {}))
        instructions = body.get("instructions")
        if instructions:
            # Responses instructions replace previous instructions, not append.
            messages.insert(0, {"role": "developer", "content": instructions})
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        custom = {}
        tools = []
        for tool in body.get("tools") or []:
            kind = tool.get("type")
            if kind == "function":
                tools.append(
                    {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
                )
            elif kind == "custom" and tool.get("format", {}).get("type", "text") == "text":
                name = tool["name"]
                custom[name] = True
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.get("description", ""),
                            "parameters": {
                                "type": "object",
                                "properties": {"input": {"type": "string"}},
                                "required": ["input"],
                                "additionalProperties": False,
                            },
                        },
                    }
                )
            else:
                format_type = (tool.get("format") or {}).get("type")
                raise GatewayError(
                    "unsupported_tool",
                    "Only caller-executed functions and text custom tools are supported "
                    f"(received name={tool.get('name')!r}, type={kind!r}, format={format_type!r})",
                )
        for item in inputs:
            kind = item.get("type", "message")
            if kind == "message":
                content = item.get("content")
                if isinstance(content, list):
                    blocks = []
                    for block in content:
                        if block.get("type") in {"input_text", "output_text", "text"}:
                            blocks.append({"type": "text", "text": block["text"]})
                        elif block.get("type") == "input_image":
                            blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": block["image_url"],
                                        "detail": block.get("detail", "low"),
                                    },
                                }
                            )
                        else:
                            raise GatewayError(
                                "unsupported_content", "Unsupported Responses content"
                            )
                    content = blocks
                messages.append({"role": item.get("role", "user"), "content": content})
            elif kind in {"function_call", "custom_tool_call"}:
                import json

                arguments = (
                    item.get("arguments")
                    if kind == "function_call"
                    else json.dumps({"input": item["input"]})
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {"name": item["name"], "arguments": arguments},
                                **(
                                    {"extra_content": item["extra_content"]}
                                    if item.get("extra_content")
                                    else {}
                                ),
                            }
                        ],
                    }
                )
            elif kind in {"function_call_output", "custom_tool_call_output"}:
                import json

                output = item["output"]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": output if isinstance(output, str) else json.dumps(output),
                    }
                )
            elif kind == "compaction":
                try:
                    summary = (
                        self.cipher(principal).decrypt(item["encrypted_content"].encode()).decode()
                    )
                except (InvalidToken, KeyError) as exc:
                    raise GatewayError(
                        "invalid_compaction",
                        "Compaction item cannot be read by this gateway/principal",
                    ) from exc
                messages.append({"role": "assistant", "content": summary})
            elif kind == "reasoning":
                if item.get("encrypted_content"):
                    context = self.provider_context(item, principal)
                    if context is not None:
                        for call_id, value in context.items():
                            if call_id in provider_bindings and provider_bindings[call_id] != value:
                                raise GatewayError(
                                    "provider_context_conflict", "Conflicting tool context", 409
                                )
                            provider_bindings[call_id] = value
                        continue
                    raise GatewayError(
                        "opaque_reasoning_unsupported",
                        "Provider reasoning cannot be replayed across deployments",
                    )
                # Summaries are portable; do not fabricate hidden reasoning.
                summary = "\n".join(b.get("text", "") for b in item.get("summary", []))
                if summary:
                    messages.append({"role": "assistant", "content": summary})
            else:
                raise GatewayError("unsupported_input", f"Unsupported Responses item: {kind}")
        # Flat Responses call items from one response form one assistant tool batch.
        grouped = []
        for message in messages:
            if (
                message.get("tool_calls")
                and grouped
                and grouped[-1].get("tool_calls")
                and message.get("content") is None
                and grouped[-1].get("content") is None
            ):
                grouped[-1]["tool_calls"].extend(message["tool_calls"])
            else:
                grouped.append(message)
        messages = grouped
        restore_calls(messages, provider_bindings)
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            choice = {"type": "function", "function": {"name": choice["name"]}}
        reasoning = body.get("reasoning")
        text = body.get("text") or {}
        fmt = text.get("format")
        if fmt and fmt.get("type") == "json_schema":
            fmt = {
                "type": "json_schema",
                "json_schema": {k: v for k, v in fmt.items() if k != "type"},
            }
        elif fmt and fmt.get("type") == "text":
            fmt = None
        request = ChatRequest(
            model=body.get("model", "theriad/agent"),
            messages=messages,
            stream=body.get("stream", False),
            tools=tools or None,
            tool_choice=choice,
            parallel_tool_calls=body.get("parallel_tool_calls"),
            max_completion_tokens=body.get("max_output_tokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            response_format=fmt,
            reasoning=reasoning,
            metadata=body.get("metadata"),
            x_theriad_session=body.get("x_theriad_session"),
        )
        request._instructions_injected = bool(instructions)
        request._namespace_map = namespaces
        request._provider_bindings = provider_bindings
        return request, custom

    def save(self, principal, request, result):
        history = request.messages[1:] if request._instructions_injected else request.messages
        messages = [m.model_dump(exclude_none=True) for m in history]
        message = {"role": "assistant", "content": None, "tool_calls": []}
        import json

        for item in result["output"]:
            if item["type"] == "message":
                message["content"] = "".join(c["text"] for c in item["content"])
            elif item["type"] in {"function_call", "custom_tool_call"}:
                name = next(
                    (
                        alias
                        for alias, pair in request._namespace_map.items()
                        if pair == (item.get("namespace"), item["name"])
                    ),
                    item["name"],
                )
                message["tool_calls"].append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": item.get(
                                "arguments", json.dumps({"input": item.get("input", "")})
                            ),
                        },
                        **(
                            {"extra_content": item["extra_content"]}
                            if item.get("extra_content")
                            else {}
                        ),
                    }
                )
        if not message["tool_calls"]:
            message.pop("tool_calls")
        messages.append(message)
        bindings = dict(request._provider_bindings)
        metadata = result.get("x_theriad", {})
        signed_calls = [
            {
                "call_id": c["id"],
                "name": c["function"]["name"],
                "arguments": c["function"]["arguments"],
                "extra_content": c["extra_content"],
            }
            for c in message.get("tool_calls", [])
            if c.get("extra_content")
        ]
        if signed_calls:
            bindings.update(
                bind_calls(signed_calls, {k: metadata[k] for k in ("release", "deployment")})
            )
        self.store.put(
            f"response/{principal}/{result['id']}",
            {"messages": messages, "provider_bindings": bindings},
            86400,
        )


class ResponseEvents:
    def __init__(self, model, custom=None, codec=None, principal=None):
        self.response = {
            "id": "resp_" + uuid.uuid4().hex,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
            "output": [],
            "usage": None,
            "error": None,
            "incomplete_details": None,
        }
        self.sequence = 0
        self.text_index = None
        self.calls = {}
        self.finish = None
        self.custom = custom or {}
        self.codec, self.principal = codec, principal

    def event(self, kind, **values):
        value = {"type": kind, "sequence_number": self.sequence, **values}
        self.sequence += 1
        return value

    def start(self):
        import copy

        return [
            self.event("response.created", response=copy.deepcopy(self.response)),
            self.event("response.in_progress", response=copy.deepcopy(self.response)),
        ]

    def feed(self, chunk):
        events = []
        if chunk.get("usage"):
            u = chunk["usage"]
            self.response["usage"] = {
                "input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get(
                    "total_tokens", u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                ),
                "input_tokens_details": {
                    "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                },
                "output_tokens_details": {
                    "reasoning_tokens": u.get("completion_tokens_details", {}).get(
                        "reasoning_tokens", 0
                    )
                },
            }
        if chunk.get("x_theriad"):
            self.response["x_theriad"] = chunk["x_theriad"]
        for choice in chunk.get("choices", []):
            self.finish = choice.get("finish_reason") or self.finish
            delta = choice.get("delta", choice.get("message", {}))
            text = delta.get("content")
            if text:
                if self.text_index is None:
                    self.text_index = len(self.response["output"])
                    item = {
                        "id": "msg_" + uuid.uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                    self.response["output"].append(item)
                    events.append(
                        self.event(
                            "response.output_item.added",
                            output_index=self.text_index,
                            item=dict(item),
                        )
                    )
                    item["content"] = [{"type": "output_text", "text": "", "annotations": []}]
                    events.append(
                        self.event(
                            "response.content_part.added",
                            item_id=item["id"],
                            output_index=self.text_index,
                            content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []},
                        )
                    )
                item = self.response["output"][self.text_index]
                item["content"][0]["text"] += text
                events.append(
                    self.event(
                        "response.output_text.delta",
                        item_id=item["id"],
                        output_index=self.text_index,
                        content_index=0,
                        delta=text,
                    )
                )
            for part in delta.get("tool_calls") or []:
                call_index = part.get("index", len(self.calls))
                if call_index not in self.calls:
                    index = len(self.response["output"])
                    item = {
                        "id": "fc_" + uuid.uuid4().hex,
                        "type": "function_call",
                        "call_id": part.get("id", ""),
                        "name": part.get("function", {}).get("name", ""),
                        "arguments": "",
                        "status": "in_progress",
                    }
                    self.response["output"].append(item)
                    self.calls[call_index] = index
                    events.append(
                        self.event(
                            "response.output_item.added", output_index=index, item=dict(item)
                        )
                    )
                else:
                    index = self.calls[call_index]
                    item = self.response["output"][index]
                    item["call_id"] += part.get("id", "")
                    item["name"] += part.get("function", {}).get("name", "")
                arguments = part.get("function", {}).get("arguments", "")
                if part.get("extra_content") is not None:
                    merge_extra_content(item.setdefault("extra_content", {}), part["extra_content"])
                item["arguments"] += arguments
                if arguments:
                    events.append(
                        self.event(
                            "response.function_call_arguments.delta",
                            item_id=item["id"],
                            output_index=index,
                            delta=arguments,
                        )
                    )
        return events

    def done(self):
        events = []
        signed_calls = [i for i in self.response["output"] if i.get("extra_content")]
        if signed_calls and self.codec:
            metadata = self.response.get("x_theriad", {})
            if not all(metadata.get(k) for k in ("release", "deployment")):
                raise GatewayError(
                    "upstream_protocol_error", "Provider context lacks a binding", 502
                )
            bindings = bind_calls(signed_calls, {k: metadata[k] for k in ("release", "deployment")})
            token = (
                self.codec.provider_cipher(self.principal)
                .encrypt(
                    json.dumps({"kind": CONTEXT_KIND, "calls": bindings}, sort_keys=True).encode()
                )
                .decode()
            )
            item = {
                "id": CONTEXT_ID_PREFIX + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [],
                "encrypted_content": token,
            }
            index = len(self.response["output"])
            self.response["output"].append(item)
            events.append(
                self.event("response.output_item.added", output_index=index, item=dict(item))
            )
        for index, item in enumerate(self.response["output"]):
            item["status"] = "completed"
            if item["type"] == "message":
                text = item["content"][0]["text"]
                events.append(
                    self.event(
                        "response.output_text.done",
                        item_id=item["id"],
                        output_index=index,
                        content_index=0,
                        text=text,
                    )
                )
                events.append(
                    self.event(
                        "response.content_part.done",
                        item_id=item["id"],
                        output_index=index,
                        content_index=0,
                        part=item["content"][0],
                    )
                )
            elif item["type"] != "reasoning":
                if not item["call_id"] or not item["name"]:
                    raise GatewayError(
                        "upstream_protocol_error", "Incomplete upstream tool call", 502
                    )
                try:
                    json.loads(item["arguments"])
                except ValueError as exc:
                    raise GatewayError(
                        "upstream_protocol_error", "Tool arguments are not valid JSON", 502
                    ) from exc
                events.append(
                    self.event(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=index,
                        arguments=item["arguments"],
                    )
                )
            events.append(self.event("response.output_item.done", output_index=index, item=item))
        self.response["status"] = (
            "incomplete" if self.finish in {"length", "content_filter"} else "completed"
        )
        if self.finish in {"length", "content_filter"}:
            self.response["incomplete_details"] = {
                "reason": "max_output_tokens" if self.finish == "length" else "content_filter"
            }
        events.append(self.event("response." + self.response["status"], response=self.response))
        return events
