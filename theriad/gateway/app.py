from __future__ import annotations

import hashlib
import hmac
import json
import os
from contextlib import aclosing

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .contracts import ChatRequest, GatewayError, ResponsesRequest
from .engine import GatewayEngine
from .releases import Releases
from .responses import ResponseEvents, ResponsesCodec
from .store import LocalStore
from .transport import WireTransport


def create_app(engine=None, tokens=None):
    app = FastAPI(title="Theriad inference gateway", version="1")
    app.state.engine = engine
    app.state.tokens = tokens

    @app.exception_handler(GatewayError)
    async def error(_request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"message": exc.message, "type": "gateway_error", "code": exc.code}},
        )

    @app.exception_handler(ValidationError)
    @app.exception_handler(RequestValidationError)
    async def invalid(_request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Request does not satisfy the gateway contract",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
        )

    def get_engine():
        if app.state.engine is None:
            if os.environ.get("THERIAD_GATEWAY_STORE") == "gcp":
                from .gcp import GoogleStore

                store = GoogleStore(
                    os.environ["GOOGLE_CLOUD_PROJECT"], os.environ["THERIAD_GATEWAY_BUCKET"]
                )
            else:
                store = LocalStore(
                    os.environ.get("THERIAD_GATEWAY_DB", "~/.theriad/gateway/state.sqlite")
                )
            key = os.environ.get("THERIAD_RELEASE_SIGNING_KEY", "")
            if not key:
                raise GatewayError("gateway_not_configured", "Release signing key is required", 503)
            releases = Releases(store, key)
            from .ports import evaluation_store

            app.state.engine = GatewayEngine(releases, WireTransport(), evaluation_store(store))
        return app.state.engine

    @app.exception_handler(json.JSONDecodeError)
    async def invalid_json(_request, _exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": {"code": "invalid_json", "message": "Request body must be valid JSON"}
            },
        )

    def principal(authorization, request=None):
        from .ports import LOOPBACK_BYOK

        LOOPBACK_BYOK.set(None)
        token = (authorization or "").removeprefix("Bearer ")
        configured = app.state.tokens
        if configured is None:
            configured = json.loads(os.environ.get("THERIAD_GATEWAY_TOKENS", "{}"))
        # Map configured SHA-256 credential hashes to principal IDs. Never store raw tokens.
        hashed = hashlib.sha256(token.encode()).hexdigest()
        for credential, owner in configured.items():
            if token and hmac.compare_digest(hashed, credential):
                if isinstance(owner, dict):
                    identity = owner.get("principal")
                    if not isinstance(identity, str) or not identity:
                        raise GatewayError(
                            "gateway_not_configured", "Credential principal is missing", 503
                        )
                    scope = {k: owner[k] for k in ("deployments", "models") if k in owner}
                    if any(
                        not isinstance(value, list)
                        or any(not isinstance(item, str) for item in value)
                        for value in scope.values()
                    ):
                        raise GatewayError(
                            "gateway_not_configured", "Credential scopes must be arrays", 503
                        )
                    from .ports import CREDENTIAL_SCOPE

                    CREDENTIAL_SCOPE.set((identity, scope))
                    return identity
                from .ports import CREDENTIAL_SCOPE

                CREDENTIAL_SCOPE.set((owner, {}))
                return owner
        if (
            os.environ.get("THERIAD_GATEWAY_LOOPBACK_BYOK", "false").lower() in {"1", "true", "yes"}
            and request is not None
            and request.client
            and request.client.host in {"127.0.0.1", "::1"}
            and token
        ):
            from .ports import CREDENTIAL_SCOPE

            owner = "loopback-byok:" + hashed
            LOOPBACK_BYOK.set(token)
            CREDENTIAL_SCOPE.set((owner, {"providers": ["openrouter"]}))
            return owner
        raise GatewayError(
            "invalid_api_key", "A scoped Theriad gateway credential is required", 401
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "theriad-gateway"}

    @app.get("/ready")
    async def ready():
        engine = get_engine()
        pointer, release = engine.releases.active()
        if not release["deployments"]:
            raise GatewayError("no_eligible_model", "Release has no deployments", 503)
        engine.select(
            "readiness",
            ChatRequest(
                model="theriad", messages=[{"role": "user", "content": "readiness"}], max_tokens=1
            ),
        )
        return {
            "ready": True,
            "release": pointer["release"],
            "deployments": len(release["deployments"]),
        }

    @app.get("/v1/models")
    async def models(request: Request, authorization: str | None = Header(default=None)):
        principal(authorization, request)
        # Codex sends client_version when it asks for its model catalog. Its
        # catalog contract is distinct from the OpenAI-compatible /models
        # response below. Keep both shapes on the same route so ordinary SDKs
        # continue to receive the standard {object, data} envelope.
        if request.query_params.get("client_version"):
            instructions = (
                "You are Codex, an AI coding assistant. Follow the user's request, "
                "use available tools when they help, preserve unrelated work, and "
                "report what you changed and verified."
            )
            return {
                "models": [
                    {
                        "slug": "theriad/agent",
                        "display_name": "Theriad",
                        "description": "Theriad quality-aware model routing",
                        "default_reasoning_level": "low",
                        "supported_reasoning_levels": [
                            {"effort": "low", "description": "Lower latency"}
                        ],
                        "shell_type": "unified_exec",
                        "visibility": "list",
                        "supported_in_api": True,
                        "priority": 1,
                        "additional_speed_tiers": [],
                        "service_tiers": [],
                        "available_access_programs": {},
                        "upgrade": None,
                        "base_instructions": instructions,
                        "model_messages": {"instructions_template": instructions},
                        "include_skills_usage_instructions": False,
                        "include_plugin_usage_instructions": False,
                        "include_apps_usage_instructions": False,
                        "default_reasoning_summary": "none",
                        "support_verbosity": False,
                        "default_verbosity": "low",
                        "apply_patch_tool_type": None,
                        "web_search_tool_type": "text",
                        "truncation_policy": {"mode": "tokens", "limit": 64000},
                        "supports_parallel_tool_calls": False,
                        "supports_image_detail_original": False,
                        "context_window": 64000,
                        "max_context_window": 64000,
                        "comp_hash": "theriad-gateway-v1",
                        "effective_context_window_percent": 95,
                        "experimental_supported_tools": [],
                        "input_modalities": ["text"],
                        "supports_search_tool": False,
                        "use_responses_lite": False,
                        "supports_reasoning_effort_updates": False,
                        "node_repl_auto_review_required": False,
                        "node_repl_disabled": True,
                        "tool_mode": "direct",
                        "multi_agent_version": None,
                    }
                ]
            }
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "theriad"}
                for name in [
                    "theriad",
                    "theriad/auto",
                    "theriad/agent",
                    "theriad/fast",
                    "theriad/quality",
                    "theriad/free",
                    "free_or_floor",
                    "cheap_bulk",
                    "cheap_code",
                    "code_strong",
                    "strong_reasoning",
                    "premium",
                    "frontier",
                ]
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(
        body: ChatRequest, request: Request, authorization: str | None = Header(default=None)
    ):
        owner = principal(authorization, request)
        engine = get_engine()
        if not body.stream:
            return await engine.complete(owner, body)
        engine.select(owner, body)

        async def events():
            try:
                async with aclosing(engine.stream(owner, body)) as upstream:
                    async for chunk in upstream:
                        if await request.is_disconnected():
                            return
                        yield "data: " + json.dumps(chunk) + "\n\n"
                yield "data: [DONE]\n\n"
            except GatewayError as exc:
                yield (
                    "data: "
                    + json.dumps({"error": {"code": exc.code, "message": exc.message}})
                    + "\n\n"
                )

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/v1/responses")
    async def responses(request: Request, authorization: str | None = Header(default=None)):
        owner = principal(authorization, request)
        engine = get_engine()
        body = ResponsesRequest.model_validate(await request.json()).model_dump(
            exclude_unset=True, exclude_none=True
        )
        codec = ResponsesCodec(engine.releases.key.decode(), engine.store)
        from .native import NativeCodec

        native_codec = NativeCodec(engine.releases.key.decode(), engine.store)
        native_request = native_codec.prepare(body, owner)
        use_native = False
        try:
            engine.select(owner, native_request)
            use_native = native_request._native_body is not None
        except GatewayError as exc:
            if exc.code not in {"no_eligible_model", "no_measured_model"}:
                raise
        if use_native:
            if not native_request.stream:
                result = None
                async for value in engine.native(owner, native_request):
                    result = value
                native_codec.restore_names(native_request, result)
                if body.get("store", True):
                    native_codec.save_native(owner, native_request, result)
                return result

            async def native_events():
                try:
                    async with aclosing(engine.native(owner, native_request)) as upstream:
                        async for event in upstream:
                            native_codec.restore_names(native_request, event)
                            if await request.is_disconnected():
                                return
                            if event["type"] in {
                                "response.completed",
                                "response.incomplete",
                            } and body.get("store", True):
                                native_codec.save_native(owner, native_request, event["response"])
                            yield (
                                "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
                            )
                except GatewayError as exc:
                    yield (
                        "event: response.failed\ndata: "
                        + json.dumps(
                            {
                                "type": "response.failed",
                                "response": {
                                    "status": "failed",
                                    "error": {"code": exc.code, "message": exc.message},
                                },
                            }
                        )
                        + "\n\n"
                    )

            return StreamingResponse(
                native_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        try:
            chat_request, custom = codec.parse(body, owner)
        except ValidationError as exc:
            raise GatewayError("invalid_request", "Invalid Responses request", 422) from exc
        if custom:
            # Do not emit fake grammar/custom-tool stream events through a function wire.
            raise GatewayError(
                "unsupported_tool",
                "Custom tools require a native Responses deployment; use function tools",
            )
        accumulator = ResponseEvents(chat_request.model, codec=codec, principal=owner)
        if not chat_request.stream:
            result = await engine.complete(owner, chat_request)
            accumulator.feed(result)
            accumulator.done()
            native_codec.restore_names(chat_request, accumulator.response)
            if body.get("store", True):
                codec.save(owner, chat_request, accumulator.response)
            return accumulator.response
        engine.select(owner, chat_request)

        async def events():
            def encode(event):
                native_codec.restore_names(chat_request, event)
                return "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"

            try:
                for event in accumulator.start():
                    yield encode(event)
                async with aclosing(engine.stream(owner, chat_request)) as upstream:
                    async for chunk in upstream:
                        if await request.is_disconnected():
                            return
                        for event in accumulator.feed(chunk):
                            yield encode(event)
                final = accumulator.done()
                if body.get("store", True):
                    codec.save(owner, chat_request, accumulator.response)
                for event in final:
                    yield encode(event)
            except GatewayError as exc:
                accumulator.response.update(
                    status="failed", error={"code": exc.code, "message": exc.message}
                )
                yield encode(accumulator.event("response.failed", response=accumulator.response))

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/v1/responses/compact")
    async def compact(request: Request, authorization: str | None = Header(default=None)):
        owner = principal(authorization, request)
        engine = get_engine()
        body = ResponsesRequest.model_validate(await request.json()).model_dump(
            exclude_unset=True, exclude_none=True
        )
        codec = ResponsesCodec(engine.releases.key.decode(), engine.store)
        from .native import NativeCodec

        native_codec = NativeCodec(engine.releases.key.decode(), engine.store)
        native_request = native_codec.prepare({**body, "stream": False, "store": False}, owner)
        directive = "Summarize the conversation for continuation. Preserve goals, constraints, tool results and unresolved work. Return only the summary."
        try:
            selection = engine.select(owner, native_request)
        except GatewayError as exc:
            if exc.code != "no_eligible_model":
                raise
            selection = None
        if selection and native_request._native_body is not None:
            native_request._native_body["input"].append({"role": "user", "content": directive})
            native_request._native_body["tools"] = []
            native_request._native_body.pop("tool_choice", None)
            native_request._native_custom = False
            # The routing projection includes the summary instruction's reserve.
            from .contracts import ChatMessage

            native_request.messages.append(ChatMessage(role="user", content=directive))
            response = None
            async for value in engine.native(owner, native_request):
                response = value
            summary = "".join(
                block["text"]
                for item in response["output"]
                if item["type"] == "message"
                for block in item.get("content", [])
                if block.get("type") == "output_text"
            )
            if not summary:
                raise GatewayError(
                    "compaction_failed", "Native deployment returned no summary", 502
                )
            import uuid

            return {
                "id": "compact_" + uuid.uuid4().hex,
                "object": "response.compaction",
                "output": [
                    {
                        "type": "compaction",
                        "id": "cmp_" + uuid.uuid4().hex,
                        "encrypted_content": codec.cipher(owner).encrypt(summary.encode()).decode(),
                    }
                ],
                "usage": response.get("usage"),
                "x_theriad": response["x_theriad"],
            }
        chat_request, _ = codec.parse({k: v for k, v in body.items() if k != "stream"}, owner)
        chat_request.messages.append(
            __import__("theriad.gateway.contracts", fromlist=["ChatMessage"]).ChatMessage(
                role="user",
                content="Summarize the conversation for continuation. Preserve goals, constraints, tool results and unresolved work. Return only the summary.",
            )
        )
        chat_request.stream = False
        chat_request.tools = None
        result = await engine.complete(owner, chat_request)
        summary = result["choices"][0]["message"].get("content")
        if not summary:
            raise GatewayError("compaction_failed", "Model returned no summary", 502)
        token = codec.cipher(owner).encrypt(summary.encode()).decode()
        return {
            "id": "compact_" + __import__("uuid").uuid4().hex,
            "object": "response.compaction",
            "output": [
                {
                    "type": "compaction",
                    "id": "cmp_" + __import__("uuid").uuid4().hex,
                    "encrypted_content": token,
                }
            ],
            "usage": result.get("usage"),
            "x_theriad": result["x_theriad"],
        }

    @app.post("/v1/theriad/episodes/{episode}/outcome")
    async def outcome(
        episode: str, request: Request, authorization: str | None = Header(default=None)
    ):
        owner = principal(authorization, request)
        key = f"episode/{owner}/{episode}"
        store = get_engine().store
        if not store.get(key):
            raise GatewayError("episode_not_found", "Episode not found", 404)
        body = await request.json()
        if type(body.get("success")) is not bool:
            raise GatewayError("invalid_outcome", "success must be boolean")

        def change(old):
            if old.get("outcome") is not None and old["outcome"] != body:
                raise GatewayError(
                    "outcome_conflict", "An episode outcome is already recorded", 409
                )
            return {**old, "outcome": body}

        store.mutate(key, change)
        return {"recorded": True}

    return app


app = create_app()
