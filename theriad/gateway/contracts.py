from __future__ import annotations

import math
import time
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


def managed_google_endpoint(project: str, location: str) -> str:
    host = (
        f"aiplatform.{location}.rep.googleapis.com"
        if location in {"us", "eu"}
        else f"{location}-aiplatform.googleapis.com"
    )
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


class GatewayError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    extra_content: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    _native_body: dict | None = PrivateAttr(default=None)
    _native_custom: bool = PrivateAttr(default=False)
    _deployment_pin: dict | None = PrivateAttr(default=None)
    _provider_bindings: dict = PrivateAttr(default_factory=dict)
    _namespace_map: dict = PrivateAttr(default_factory=dict)
    _instructions_injected: bool = PrivateAttr(default=False)
    model_config = ConfigDict(extra="forbid")
    model: str = "theriad/agent"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    user: str | None = None
    store: bool = False
    metadata: dict[str, str] | None = None
    x_theriad_session: str | None = None
    x_theriad_policy: str = "balanced"
    x_theriad_speed: int | None = Field(default=None, ge=0, le=100)
    x_theriad_quality: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def validate_tools(self):
        if (
            self.x_theriad_policy != "balanced"
            or self.x_theriad_speed is not None
            or self.x_theriad_quality is not None
        ):
            raise ValueError(
                "gateway supports the frozen balanced routing policy; per-request preference overrides are not implemented"
            )
        for tool in self.tools or []:
            if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ValueError("only caller-executed function tools are supported")
        for message in self.messages:
            if message.role == "tool" and not message.tool_call_id:
                raise ValueError("tool results require tool_call_id")
        return self

    @property
    def output_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 4096

    def wire(self, model: str) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        data["messages"] = [
            m.model_dump(exclude_none=True)
            | ({"content": None} if m.content is None and "content" in m.model_fields_set else {})
            for m in self.messages
        ]
        for key in tuple(data):
            if key.startswith("x_theriad_") or key in {"metadata", "store"}:
                data.pop(key)
        data["model"] = model
        if "max_tokens" not in data and "max_completion_tokens" not in data:
            data["max_tokens"] = self.output_limit
        if self.stream:
            data["stream_options"] = {"include_usage": True}
        return data


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "theriad/agent"
    input: str | list[dict[str, Any]] = Field(default_factory=list)
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = True
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    include: list[str] | None = None
    truncation: str | None = None
    prompt_cache_key: str | None = None
    service_tier: str | None = None
    background: bool = False
    x_theriad_session: str | None = None
    client_metadata: dict[str, Any] | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ResponsesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    status: Literal["in_progress", "completed", "incomplete", "failed", "queued"]
    output: list[dict[str, Any]]
    usage: Usage | None = None


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = Field(min_length=1)
    sequence_number: int | None = Field(default=None, ge=0)
    response: ResponsesResponse | None = None


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    model: str
    revision: str
    provider: Literal["openrouter", "openai", "compatible", "vertex"]
    endpoint: str
    region: str
    tiers: list[str] = Field(min_length=1)
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    tools: bool = False
    vision: bool = False
    reasoning: bool = False
    schemas: bool = False
    input_price: float = Field(ge=0, allow_inf_nan=False)
    output_price: float = Field(ge=0, allow_inf_nan=False)
    cache_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    priced_at: float
    credential_env: str | None = None
    project: str | None = None
    quality_prior: float = Field(default=0.5, ge=0, le=1)
    origin: str | None = None
    serving_variant: str = "default"
    reasoning_profile: str = "default"
    reasoning_config: dict = Field(default_factory=dict)
    hosting_provider: str | None = None
    supported_parameters: list[str] | None = None
    request_price: float = Field(default=0, ge=0, allow_inf_nan=False)
    canary_cost_ratio: float = Field(default=1.05, ge=1, allow_inf_nan=False)
    canary_latency_ratio: float = Field(default=1.10, ge=1, allow_inf_nan=False)
    canary_failure_delta: float = Field(default=0.02, ge=0, le=1, allow_inf_nan=False)
    native_responses: bool = False
    custom_tools: bool = False

    @model_validator(mode="after")
    def endpoint_valid(self):
        from theriad.models.tiers import VALID_TIERS

        if set(self.tiers) - VALID_TIERS:
            raise ValueError("unknown escalation tier")
        url = urlsplit(self.endpoint)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("endpoint cannot contain credentials/query/fragment")
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("TLS is required except for loopback endpoints")
        if (
            self.provider == "openrouter"
            and self.endpoint.rstrip("/") != "https://openrouter.ai/api/v1"
        ):
            raise ValueError("OpenRouter credentials require the canonical OpenRouter endpoint")
        if self.provider == "openai" and self.endpoint.rstrip("/") != "https://api.openai.com/v1":
            raise ValueError("OpenAI credentials require the canonical OpenAI endpoint")
        if self.provider == "vertex":
            if not self.project or "/" in self.project:
                raise ValueError("vertex requires a concrete project")
            import re

            if self.region not in {"us", "eu"} and not re.fullmatch(
                r"[a-z]+-[a-z]+[0-9]+", self.region
            ):
                raise ValueError("vertex requires an explicit Google region or us/eu multi-region")
            expected = managed_google_endpoint(self.project, self.region)
            if self.endpoint.rstrip("/") != expected:
                raise ValueError("Google credentials require the declared managed Vertex endpoint")
        return self

    def maximum_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            self.request_price
            + (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000
        )


class EndpointPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_origins: list[str] | None = None
    allowed_endpoints: list[str]
    allowed_regions: list[str]
    excluded_origins: list[str] = Field(default_factory=list)
    max_price_age_seconds: int = 86400
    escalation_ceiling: Literal[
        "free_or_floor",
        "cheap_bulk",
        "cheap_code",
        "code_strong",
        "strong_reasoning",
        "premium",
        "frontier",
    ] = "premium"

    def allows(self, deployment: Deployment) -> bool:
        return (
            deployment.endpoint in self.allowed_endpoints
            and deployment.region in self.allowed_regions
            and deployment.origin not in self.excluded_origins
            and (not self.excluded_origins or deployment.origin is not None)
            and (self.allowed_origins is None or deployment.origin in self.allowed_origins)
        )


def context_size(request: ChatRequest) -> int:
    """Conservative UTF-8 byte bound, including complete tool schemas and reserve.

    Bytes bound text token counts without assuming English or a specific tokenizer.
    Images must have a documented conservative bound; unknown blocks fail closed.
    """
    import json

    size = len(json.dumps(request.model_dump(exclude_none=True), ensure_ascii=False).encode())
    for message in request.messages:
        for block in message.content if isinstance(message.content, list) else []:
            kind = block.get("type")
            if kind == "image_url":
                # Provider image tokenization depends on resolution. Reserve enough
                # for supported bounded image inputs; don't fetch arbitrary URLs.
                value = block.get("image_url", {})
                if not isinstance(value, dict) or value.get("detail", "auto") != "low":
                    raise GatewayError(
                        "unsupported_image_detail", "Gateway requires image detail=low"
                    )
                size += 4096
            elif kind != "text":
                raise GatewayError("unsupported_content", f"Unsupported content block: {kind}")
    return size + request.output_limit


def vertex_compatible(deployment: Deployment, request: ChatRequest) -> bool:
    if deployment.provider != "vertex":
        return True
    reasoning = request.reasoning or {}
    effort = (
        request.reasoning_effort
        or reasoning.get("effort")
        or deployment.reasoning_config.get("effort")
    )
    if effort is not None and effort not in {"low", "medium", "high"}:
        return False
    if set(reasoning) - {"effort", "summary"} or reasoning.get("summary") not in {
        None,
        "none",
        "auto",
    }:
        return False
    if any(m.reasoning or m.reasoning_details for m in request.messages):
        return False

    # Fully recursive schemas are unsupported. Reject references conservatively
    # before dispatch rather than silently removing schema constraints.
    def references(value):
        if isinstance(value, dict):
            return "$ref" in value or any(references(v) for v in value.values())
        return isinstance(value, list) and any(references(v) for v in value)

    return not references(request.response_format) and not references(request.tools)


def billed_output_bound(deployment: Deployment, request: ChatRequest) -> int:
    if deployment.provider != "vertex":
        return request.output_limit
    effort = (
        request.reasoning_effort
        or (request.reasoning or {}).get("effort")
        or deployment.reasoning_config.get("effort")
    )
    budget = {"low": 1024, "medium": 8192, "high": 24576}.get(effort)
    # Reserve reasoning as well as visible output. If reasoning configuration is
    # unspecified, use the deployment's entire documented output ceiling.
    return (
        min(deployment.max_output_tokens, request.output_limit + budget)
        if budget is not None
        else deployment.max_output_tokens
    )


def eligible(
    deployment: Deployment, request: ChatRequest, policy: EndpointPolicy, now: float | None = None
) -> bool:
    now = time.time() if now is None else now
    images = any(
        isinstance(m.content, list) and any(b.get("type") == "image_url" for b in m.content)
        for m in request.messages
    )
    reasoning = bool(
        any(
            k not in {"effort", "summary"} or v != "none"
            for k, v in (request.reasoning or {}).items()
        )
        or request.reasoning_effort not in {None, "none"}
        or any(m.reasoning_details for m in request.messages)
    )
    requested_effort = request.reasoning_effort or (request.reasoning or {}).get("effort")
    fixed_effort = deployment.reasoning_config.get("effort")
    order = [
        "free_or_floor",
        "cheap_bulk",
        "cheap_code",
        "code_strong",
        "strong_reasoning",
        "premium",
        "frontier",
    ]
    ceiling = order.index(policy.escalation_ceiling)
    permitted_tier = any(
        t.startswith("free_moe_") or t in order[: ceiling + 1] for t in deployment.tiers
    )
    return (
        policy.allows(deployment)
        and vertex_compatible(deployment, request)
        and (deployment.provider != "openrouter" or bool(deployment.hosting_provider))
        and permitted_tier
        and math.isfinite(deployment.priced_at)
        and 0 <= now - deployment.priced_at <= policy.max_price_age_seconds
        and deployment.context_window >= context_size(request)
        and (
            deployment.max_input_tokens is None
            or context_size(request) - request.output_limit <= deployment.max_input_tokens
        )
        and deployment.max_output_tokens >= request.output_limit
        and (
            request.model != "theriad/agent"
            or (deployment.tools and deployment.context_window >= 64000)
        )
        and (not request.tools or deployment.tools)
        and (
            request.parallel_tool_calls is None
            or deployment.supported_parameters is None
            or "parallel_tool_calls" in deployment.supported_parameters
        )
        and (not images or deployment.vision)
        and (not reasoning or deployment.reasoning)
        and (not fixed_effort or not requested_effort or fixed_effort == requested_effort)
        and (request._native_body is None or deployment.native_responses)
        and (not request._native_custom or deployment.custom_tools)
        and (not request.response_format or deployment.schemas)
    )
