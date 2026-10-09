"""Actual upstream Chat Completions/SSE, bypassing all prompt transformations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError

from .contracts import (
    ChatRequest,
    Deployment,
    GatewayError,
    ResponsesResponse,
    StreamEvent,
    Usage,
    vertex_compatible,
)


class WireTransport:
    def __init__(self, transport=None, credentials=None, credential_resolver=None):
        self.transport = transport
        from .ports import LocalAndGoogleCredentials

        self.credential_resolver = credential_resolver or LocalAndGoogleCredentials(credentials)

    @staticmethod
    def body(deployment, request):
        body = (
            {
                **request._native_body,
                "model": deployment.model,
                "stream": request.stream,
                "store": False,
            }
            if request._native_body is not None
            else request.wire(deployment.model)
        )
        if request._native_body is not None:
            body.setdefault("max_output_tokens", request.output_limit)
        if not deployment.reasoning:
            # An explicit no-reasoning request has the same semantics on an
            # endpoint that implements no reasoning mode. Avoid sending a
            # provider-specific parameter it cannot accept.
            if body.get("reasoning_effort") == "none":
                body.pop("reasoning_effort")
            if body.get("reasoning") and all(
                k in {"effort", "summary"} and v == "none" for k, v in body["reasoning"].items()
            ):
                body.pop("reasoning")
        if deployment.reasoning_config:
            body["reasoning"] = {**deployment.reasoning_config, **(body.get("reasoning") or {})}
        if (
            body.get("parallel_tool_calls") is False
            and deployment.supported_parameters is not None
            and "parallel_tool_calls" not in deployment.supported_parameters
        ):
            # A disabled parallel-call hint does not require an unsupported
            # parameter to be sent upstream; function calls remain usable.
            body.pop("parallel_tool_calls")
        if deployment.provider == "vertex":
            if not vertex_compatible(deployment, request):
                raise GatewayError(
                    "unsupported_parameter",
                    "Google deployment cannot preserve request settings",
                    422,
                )
            reasoning = body.pop("reasoning", {}) or {}
            effort = body.get("reasoning_effort") or reasoning.get("effort")
            if effort:
                body["reasoning_effort"] = effort
            # Single-call requests are enforced by the gateway response validator;
            # this parameter is not supported by Google's upstream wire.
            body.pop("parallel_tool_calls", None)
            for message in body["messages"]:
                if message["role"] == "developer":
                    message["role"] = "system"
        for message in body.get("messages", []):
            for call in message.get("tool_calls") or []:
                binding = request._provider_bindings.get(call["id"])
                if binding and binding["deployment"] != deployment.id:
                    call.pop("extra_content", None)
        if deployment.provider == "openrouter":
            body["provider"] = {
                "require_parameters": True,
                "max_price": {
                    "prompt": deployment.input_price,
                    "completion": deployment.output_price,
                    "request": deployment.request_price,
                    "image": 0,
                },
            }
            if deployment.hosting_provider:
                body["provider"].update(only=[deployment.hosting_provider], allow_fallbacks=False)
        return body

    async def charge(self, deployment, generation_id):
        if deployment.provider != "openrouter" or not generation_id:
            return None
        async with httpx.AsyncClient(transport=self.transport, timeout=30) as client:
            response = await client.get(
                deployment.endpoint.rstrip("/") + "/generation",
                params={"id": generation_id},
                headers=await self.headers(deployment),
            )
            if response.status_code != 200:
                return None
            data = response.json().get("data", {})
            if data.get("id") != generation_id or data.get("model") != deployment.model:
                raise GatewayError(
                    "charge_identity_mismatch",
                    "Provider charge identity does not match evaluation",
                    409,
                )
            return data.get("total_cost")

    async def headers(self, deployment: Deployment) -> dict[str, str]:
        token = await self.credential_resolver.token(deployment)
        if not token:
            raise GatewayError(
                "provider_credentials_missing", "Deployment credential is unavailable", 503
            )
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def complete(self, deployment: Deployment, request: ChatRequest) -> dict:
        headers = await self.headers(deployment)
        async with httpx.AsyncClient(
            transport=self.transport, timeout=httpx.Timeout(600, read=120)
        ) as client:
            try:
                response = await client.post(
                    deployment.endpoint.rstrip("/") + "/chat/completions",
                    headers=headers,
                    json=self.body(deployment, request),
                )
            except httpx.HTTPError as exc:
                raise GatewayError("upstream_error", "Upstream connection failed", 502) from exc
            if response.status_code != 200:
                self.reject(response.status_code)
            try:
                data = response.json()
            except ValueError as exc:
                raise GatewayError("upstream_protocol_error", "Invalid upstream JSON", 502) from exc
            self.validate(data)
            self.normalize_usage(data, deployment)
            if len(data["choices"]) != 1 or not data["choices"][0].get("finish_reason"):
                raise GatewayError(
                    "upstream_protocol_error", "Completion has no terminal choice", 502
                )
            message = data["choices"][0].get("message")
            if not isinstance(message, dict):
                raise GatewayError(
                    "upstream_protocol_error", "Completion lacks assistant message", 502
                )
            if (
                not message.get("content")
                and not message.get("tool_calls")
                and data["choices"][0]["finish_reason"] not in {"length", "content_filter"}
            ):
                raise GatewayError(
                    "upstream_protocol_error", "Completion has no text or tool calls", 502
                )
            self.validate_message(message)
            if request.parallel_tool_calls is False and len(message.get("tool_calls") or []) > 1:
                raise GatewayError(
                    "upstream_protocol_error", "Parallel calls violated the request", 502
                )
            self.validate_identity(data, deployment)
            return data

    @staticmethod
    def normalize_usage(data, deployment):
        usage = data.get("usage")
        if deployment.provider != "vertex" or not usage:
            return
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        if type(reasoning) is not int or reasoning < 0:
            raise GatewayError("upstream_protocol_error", "Invalid Google reasoning usage", 502)
        prompt, output, total = (
            usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if reasoning and any(type(v) is not int for v in (prompt, output, total)):
            raise GatewayError("upstream_protocol_error", "Incomplete Google token accounting", 502)
        if reasoning and total == prompt + output + reasoning:
            usage["provider_completion_tokens"] = output
            usage["completion_tokens"] = output + reasoning
        elif reasoning and total != prompt + output:
            raise GatewayError("upstream_protocol_error", "Ambiguous Google token accounting", 502)

    @staticmethod
    def validate_identity(data, deployment):
        if data.get("model") not in {deployment.model, deployment.revision}:
            raise GatewayError(
                "unauthorized_routing", "Upstream returned an unexpected model identity", 502
            )

    @staticmethod
    def validate_message(message):
        if not isinstance(message, dict) or (
            message.get("content") is not None and not isinstance(message["content"], str)
        ):
            raise GatewayError("upstream_protocol_error", "Malformed assistant content", 502)
        if message.get("tool_calls") is not None and not isinstance(message["tool_calls"], list):
            raise GatewayError("upstream_protocol_error", "Malformed tool call array", 502)
        ids = set()
        for call in message.get("tool_calls") or []:
            if (
                not isinstance(call, dict)
                or not call.get("id")
                or call.get("id") in ids
                or call.get("type") != "function"
                or not isinstance(call.get("function"), dict)
                or not call["function"].get("name")
            ):
                raise GatewayError(
                    "upstream_protocol_error", "Malformed or duplicate tool call", 502
                )
            ids.add(call["id"])
            try:
                value = json.loads(call["function"]["arguments"])
                if not isinstance(value, dict):
                    raise ValueError("arguments must be an object")
            except (ValueError, KeyError, TypeError) as exc:
                raise GatewayError(
                    "upstream_protocol_error", "Invalid tool arguments", 502
                ) from exc

    @staticmethod
    def validate_native(data, deployment):
        WireTransport.validate_identity(data, deployment)
        for item in data.get("output", []):
            if not isinstance(item, dict):
                raise GatewayError("upstream_protocol_error", "Malformed native output item", 502)
            kind = item.get("type")
            if kind == "function_call":
                WireTransport.validate_message(
                    {
                        "tool_calls": [
                            {
                                "id": item.get("call_id"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name"),
                                    "arguments": item.get("arguments"),
                                },
                            }
                        ]
                    }
                )
            elif kind == "custom_tool_call":
                if (
                    not deployment.custom_tools
                    or not item.get("call_id")
                    or not item.get("name")
                    or not isinstance(item.get("input"), str)
                ):
                    raise GatewayError(
                        "upstream_protocol_error", "Malformed or unauthorized custom tool call", 502
                    )
            elif kind == "message":
                if not isinstance(item.get("content"), list) or any(
                    not isinstance(b, dict)
                    or b.get("type") not in {"output_text", "refusal"}
                    or not isinstance(b.get("text", b.get("refusal")), str)
                    for b in item["content"]
                ):
                    raise GatewayError("upstream_protocol_error", "Malformed native message", 502)
            elif kind != "reasoning":
                raise GatewayError(
                    "upstream_protocol_error", "Unimplemented native output capability", 502
                )

    @staticmethod
    def reject(status):
        retryable = status in {401, 403, 408, 429} or status >= 500
        raise GatewayError(
            "upstream_error" if retryable else "upstream_rejected",
            "Upstream rejected the request",
            502 if retryable else 422,
        )

    async def responses(self, deployment, request):
        headers = await self.headers(deployment)
        body = self.body(deployment, request)
        async with httpx.AsyncClient(
            transport=self.transport, timeout=httpx.Timeout(600, read=120)
        ) as client:
            if not request.stream:
                response = await client.post(
                    deployment.endpoint.rstrip("/") + "/responses", headers=headers, json=body
                )
                if response.status_code != 200:
                    self.reject(response.status_code)
                data = response.json()
                try:
                    ResponsesResponse.model_validate(data)
                except ValidationError as exc:
                    raise GatewayError(
                        "upstream_protocol_error", "Malformed native response", 502
                    ) from exc
                if (
                    not isinstance(data, dict)
                    or data.get("status") not in {"completed", "incomplete"}
                    or not isinstance(data.get("output"), list)
                ):
                    raise GatewayError("upstream_protocol_error", "Invalid native response", 502)
                self.validate_native(data, deployment)
                yield data
                return
            async with client.stream(
                "POST", deployment.endpoint.rstrip("/") + "/responses", headers=headers, json=body
            ) as response:
                if response.status_code != 200:
                    self.reject(response.status_code)
                fields = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        fields.append(line[5:].lstrip())
                    elif not line and fields:
                        value, fields = "\n".join(fields), []
                        if value == "[DONE]":
                            return
                        try:
                            data = json.loads(value)
                        except ValueError as exc:
                            raise GatewayError(
                                "upstream_protocol_error", "Invalid native stream JSON", 502
                            ) from exc
                        if not isinstance(data, dict) or not isinstance(data.get("type"), str):
                            raise GatewayError(
                                "upstream_protocol_error", "Invalid native stream event", 502
                            )
                        try:
                            StreamEvent.model_validate(data)
                        except ValidationError as exc:
                            raise GatewayError(
                                "upstream_protocol_error", "Malformed native stream event", 502
                            ) from exc
                        if data["type"] in {"error", "response.failed"}:
                            raise GatewayError(
                                "upstream_protocol_error", "Upstream native stream failed", 502
                            )
                        if (data.get("response") or {}).get("status") in {
                            "completed",
                            "incomplete",
                        }:
                            self.validate_native(data["response"], deployment)
                        yield data
                if fields:
                    raise GatewayError("upstream_protocol_error", "Truncated native event", 502)

    @staticmethod
    def validate(data):
        if (
            not isinstance(data, dict)
            or data.get("error")
            or not isinstance(data.get("choices"), list)
        ):
            raise GatewayError("upstream_protocol_error", "Invalid upstream completion", 502)
        if len(data["choices"]) > 1:
            raise GatewayError(
                "upstream_protocol_error", "Only one completion choice is supported", 502
            )
        if data["choices"]:
            choice = data["choices"][0]
            if not isinstance(choice, dict) or not isinstance(
                choice.get("message", choice.get("delta")), dict
            ):
                raise GatewayError("upstream_protocol_error", "Invalid upstream choice", 502)
        if data.get("usage"):
            try:
                Usage.model_validate(data["usage"])
            except ValidationError as exc:
                raise GatewayError(
                    "upstream_protocol_error", "Invalid upstream usage", 502
                ) from exc

    async def stream(self, deployment: Deployment, request: ChatRequest) -> AsyncIterator[dict]:
        headers = await self.headers(deployment)
        call_indices = set()
        async with httpx.AsyncClient(
            transport=self.transport, timeout=httpx.Timeout(600, read=120)
        ) as client:
            async with client.stream(
                "POST",
                deployment.endpoint.rstrip("/") + "/chat/completions",
                headers=headers,
                json=self.body(deployment, request),
            ) as response:
                if response.status_code != 200:
                    self.reject(response.status_code)
                fields = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        fields.append(line[5:].lstrip())
                    elif not line and fields:
                        value = "\n".join(fields)
                        fields = []
                        if value == "[DONE]":
                            return
                        try:
                            data = json.loads(value)
                        except ValueError as exc:
                            raise GatewayError(
                                "upstream_protocol_error", "Invalid upstream stream", 502
                            ) from exc
                        self.validate(data)
                        self.normalize_usage(data, deployment)
                        if data.get("model"):
                            self.validate_identity(data, deployment)
                        for choice in data["choices"]:
                            delta = choice.get("delta", {})
                            if not isinstance(delta, dict) or (
                                delta.get("tool_calls") is not None
                                and not isinstance(delta["tool_calls"], list)
                            ):
                                raise GatewayError(
                                    "upstream_protocol_error", "Malformed streaming delta", 502
                                )
                            if delta.get("content") is not None and not isinstance(
                                delta["content"], str
                            ):
                                raise GatewayError(
                                    "upstream_protocol_error", "Malformed text delta", 502
                                )
                            for call in delta.get("tool_calls") or []:
                                if (
                                    not isinstance(call, dict)
                                    or type(call.get("index")) is not int
                                    or call.get("index", -1) < 0
                                    or not isinstance(call.get("function", {}), dict)
                                ):
                                    raise GatewayError(
                                        "upstream_protocol_error", "Malformed tool delta", 502
                                    )
                                call_indices.add(call["index"])
                                if request.parallel_tool_calls is False and len(call_indices) > 1:
                                    raise GatewayError(
                                        "upstream_protocol_error",
                                        "Parallel calls violated the request",
                                        502,
                                    )
                                if (
                                    call.get("id") is not None and not isinstance(call["id"], str)
                                ) or any(
                                    v is not None and not isinstance(v, str)
                                    for k, v in call.get("function", {}).items()
                                    if k in {"name", "arguments"}
                                ):
                                    raise GatewayError(
                                        "upstream_protocol_error",
                                        "Malformed tool delta values",
                                        502,
                                    )
                        yield data
                if fields:
                    raise GatewayError("upstream_protocol_error", "Truncated upstream event", 502)
