from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from contextlib import aclosing
from dataclasses import dataclass

import httpx

from .contracts import (
    ChatRequest,
    Deployment,
    EndpointPolicy,
    GatewayError,
    billed_output_bound,
    context_size,
    eligible,
)
from .releases import digest


@dataclass
class Selection:
    release: str
    deployments: list[Deployment]
    session_key: str
    policy: EndpointPolicy
    principal: str


class GatewayEngine:
    def __init__(self, releases, transport, accounting_store=None, audit=None):
        self.releases, self.store, self.transport = releases, releases.store, transport
        self.accounting_store = accounting_store or self.store
        from .ports import StoreAudit

        self.audit = audit or StoreAudit(self.store)
        self.scopes = {}
        self.routers = {}

    def scope_for(self, principal):
        from .ports import CREDENTIAL_SCOPE

        scoped = CREDENTIAL_SCOPE.get()
        return scoped[1] if scoped and scoped[0] == principal else self.scopes.get(principal, {})

    def reserve_attempt(self, principal, deployment, request):
        if not principal.startswith("eval:"):
            return None
        from .adoption import MonthlyBudget

        return MonthlyBudget(self.accounting_store).reserve(
            deployment.maximum_cost(
                context_size(request), billed_output_bound(deployment, request)
            ),
            "qualification" if principal.startswith("eval:qualification:") else "pilot",
            "harness/" + uuid.uuid4().hex,
        )

    def settle_attempt(self, reservation, completion, deployment):
        if reservation:
            from .adoption import MonthlyBudget

            budget = MonthlyBudget(self.accounting_store)
            budget.defer(reservation, deployment, completion.get("id"))
            budget.settle(*reservation, (completion.get("usage") or {}).get("cost"))

    def select(self, principal: str, request: ChatRequest) -> Selection:
        pointer, payload = self.releases.active()
        # Hash the stable instruction/tool prefix if no explicit session is provided.
        # Tool IDs, scoped to the principal, are authoritative for outstanding cycles.
        key = hashlib.sha256(
            (
                principal
                + (
                    request.x_theriad_session
                    or digest(
                        {
                            "instructions": [
                                m.model_dump(exclude_none=True)
                                for m in request.messages
                                if m.role in {"system", "developer"}
                            ],
                            "tools": request.tools,
                        }
                    )
                )
            ).encode()
        ).hexdigest()
        result_ids = []
        for message in reversed(request.messages):
            if message.role != "tool":
                break
            result_ids.append(message.tool_call_id)
        result_ids.reverse()
        pins = [self.store.get(f"call/{principal}/{call}") for call in result_ids]
        if result_ids and any(pin is None for pin in pins):
            raise GatewayError(
                "tool_cycle_unavailable",
                "Outstanding tool bindings are unknown or expired for this principal",
                409,
            )
        pins = [pin for pin in pins if pin]
        # Historical tool calls don't pin a fresh turn unless latest input is a result.
        pin = pins[-1] if request.messages[-1].role == "tool" and pins else None
        if pin and any(p != pin for p in pins):
            raise GatewayError(
                "tool_cycle_conflict",
                "Outstanding tool results span different deployment cycles",
                409,
            )
        if request._deployment_pin:
            if pin and pin != request._deployment_pin:
                raise GatewayError(
                    "tool_cycle_conflict", "Tool and opaque reasoning bindings disagree", 409
                )
            pin = request._deployment_pin
        release_id = pin["release"] if pin else pointer["release"]
        canary = pointer.get("canary")
        if release_id != pointer["release"]:
            payload = self.releases.load(release_id)
        policy = EndpointPolicy.model_validate(payload["policy"])
        # Current policy always wins over a pinned historical policy/release.
        current_policy = EndpointPolicy.model_validate(self.releases.active()[1]["policy"])
        candidates = [Deployment.model_validate(d) for d in payload["deployments"]]
        scope = self.scope_for(principal)
        candidates = [
            d
            for d in candidates
            if ("deployments" not in scope or d.id in scope["deployments"])
            and ("models" not in scope or d.model in scope["models"])
            and ("providers" not in scope or d.provider in scope["providers"])
            and eligible(d, request, policy)
            and eligible(d, request, current_policy)
        ]
        from theriad.models.tiers import VALID_TIERS

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
        candidates = [
            d
            for d in candidates
            if any(t.startswith("free_moe_") or t in order[: ceiling + 1] for t in d.tiers)
        ]
        if pin:
            candidates = [d for d in candidates if d.id == pin["deployment"]]
        elif request.model in VALID_TIERS:
            candidates = [d for d in candidates if request.model in d.tiers]
        elif request.model not in {
            "theriad",
            "theriad/auto",
            "theriad/agent",
            "theriad/fast",
            "theriad/quality",
            "theriad/free",
        }:
            candidates = [
                d for d in candidates if d.model == request.model or d.id == request.model
            ]
        elif request.model == "theriad/free":
            candidates = [d for d in candidates if d.input_price == d.output_price == 0]
        if not candidates:
            raise GatewayError(
                "no_eligible_model",
                "No authorized deployment satisfies context, capabilities, output, and pricing requirements",
                422,
            )
        prompt = "\n".join(str(m.content or "") for m in request.messages)
        candidates.sort(
            key=lambda d: (
                d.maximum_cost(context_size(request), request.output_limit),
                -d.quality_prior,
                d.id,
            )
        )
        if request.model == "theriad/quality":
            candidates.sort(key=lambda d: (-d.quality_prior, d.input_price + d.output_price))
        original_candidates = list(candidates)
        if payload.get("router") and not pin:
            from theriad.models.tiers import ModelCandidate
            from theriad.routing.v2 import RouterV2

            router = self.routers.get(release_id)
            if router is None:
                from .router_runtime import load_router

                router = load_router(payload["router"], payload)
                self.routers[release_id] = router
            rows = [
                ModelCandidate(
                    tier=d.tiers[0],
                    provider=d.provider,
                    model=d.id,
                    input_cost_per_million=d.input_price,
                    output_cost_per_million=d.output_price,
                    context_window=d.context_window,
                    quality_prior=d.quality_prior,
                )
                for d in candidates
            ]
            lane = "code" if request.tools else "general"
            explicit = (
                request.model
                not in {
                    "theriad",
                    "theriad/auto",
                    "theriad/agent",
                    "theriad/fast",
                    "theriad/quality",
                    "theriad/free",
                }
                and request.model not in VALID_TIERS
            )
            if explicit and all(d.id not in router.artifact.models for d in candidates):
                replacement = next(
                    (
                        r
                        for r in reversed(payload.get("replacements", []))
                        if lane in r["lanes"] and any(d.id == r["candidate"] for d in candidates)
                    ),
                    None,
                )
                if replacement:
                    from theriad.routing.v2 import RouterV2Artifact

                    router = RouterV2(RouterV2Artifact.from_dict(replacement["artifact"]))
            try:
                decision = router.predict(prompt, rows, lane=lane)
            except ValueError as exc:
                raise GatewayError(
                    "no_measured_model", "No eligible measured deployment is available", 422
                ) from exc
            selected_model = decision.selected_model
            measured_scores = [p for p in decision.candidate_scores if p.eligible]
            if request.model == "theriad/quality":
                selected_model = max(
                    measured_scores, key=lambda p: (p.p_success_lower, -p.expected_cost_usd)
                ).model
            elif request.model == "theriad/fast":
                fast = [
                    p
                    for p in measured_scores
                    if p.expected_latency_ms is not None
                    and p.p_success_lower >= decision.knee_threshold
                ]
                if fast:
                    selected_model = min(
                        fast, key=lambda p: (p.expected_latency_ms, p.expected_cost_usd)
                    ).model
            # Only measured/promoted predictions remain eligible; no unseen fallback.
            allowed = {p.model for p in decision.candidate_scores if p.eligible}
            candidates = [d for d in candidates if d.id in allowed]
            candidates.sort(key=lambda d: d.id != selected_model)
            # Replacement artifacts retain their own feature normalization.
            # Only replace a deployment the incumbent policy would have selected.
            for replacement in reversed(payload.get("replacements", [])):
                if (
                    lane not in replacement["lanes"]
                    or not candidates
                    or candidates[0].id != replacement["incumbent"]
                ):
                    continue
                pair = [
                    d
                    for d in original_candidates
                    if d.id in {replacement["candidate"], replacement["incumbent"]}
                ]
                if not any(d.id == replacement["candidate"] for d in pair):
                    continue
                overlay_key = (release_id, replacement["candidate"])
                overlay = self.routers.get(overlay_key)
                if overlay is None:
                    from theriad.routing.v2 import RouterV2Artifact

                    overlay = RouterV2(RouterV2Artifact.from_dict(replacement["artifact"]))
                    self.routers[overlay_key] = overlay
                try:
                    scores = overlay.predict(
                        prompt,
                        [
                            ModelCandidate(
                                tier=d.tiers[0],
                                provider=d.provider,
                                model=d.id,
                                input_cost_per_million=d.input_price,
                                output_cost_per_million=d.output_price,
                                context_window=d.context_window,
                            )
                            for d in pair
                        ],
                        lane=lane,
                    )
                except ValueError:
                    continue
                measured = {p.model for p in scores.candidate_scores if p.eligible}
                pair = [d for d in pair if d.id in measured]
                pair.sort(key=lambda d: d.id != scores.selected_model)
                candidates = pair + [d for d in candidates if d.id not in measured]
                break
        if not candidates:
            raise GatewayError("no_measured_model", "No measured deployment is available", 422)
        # Canary only the traffic whose frozen incumbent would have handled it.
        # Explicit pins and unrelated tiers/scopes remain on their chosen route.
        if (
            canary
            and not pin
            and not self.store.get("updates/paused")
            and candidates[0].id == canary["report"]["incumbent"]
            and request.model
            in {
                "theriad",
                "theriad/auto",
                "theriad/agent",
                "theriad/fast",
                "theriad/quality",
                "theriad/free",
                *VALID_TIERS,
            }
            and int(uuid.uuid4().hex[:8], 16) / 2**32 < canary["fraction"]
        ):
            candidate_payload = self.releases.load(canary["candidate"])
            candidate_policy = EndpointPolicy.model_validate(candidate_payload["policy"])
            challenger = next(
                (
                    Deployment.model_validate(d)
                    for d in candidate_payload["deployments"]
                    if d["id"] == canary["report"]["candidate"]
                ),
                None,
            )
            if (
                challenger
                and eligible(challenger, request, candidate_policy)
                and eligible(challenger, request, current_policy)
                and ("deployments" not in scope or challenger.id in scope["deployments"])
                and ("models" not in scope or challenger.model in scope["models"])
                and ("providers" not in scope or challenger.provider in scope["providers"])
                and (request.model not in VALID_TIERS or request.model in challenger.tiers)
                and (
                    request.model != "theriad/free"
                    or challenger.input_price == challenger.output_price == 0
                )
            ):
                lane = "code" if request.tools else "general"
                family_qualified = lane in canary["report"].get("families", {}) or lane in canary[
                    "report"
                ].get("qualified_lanes", [])
                if family_qualified:
                    # No control completion is mislabeled as a candidate fallback.
                    return Selection(
                        canary["candidate"], [challenger], key, candidate_policy, principal
                    )
        return Selection(release_id, candidates, key, policy, principal)

    def record(self, principal, selection, deployment, completion, latency_ms=None):
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            self.store.put(
                f"call/{principal}/{call['id']}",
                {"release": selection.release, "deployment": deployment.id},
                86400,
            )
        usage = completion.get("usage")
        cost = None
        reported_cost = (usage or {}).get("cost")
        if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
            cost = usage.get("cost")
            if cost is None:
                cost = deployment.maximum_cost(usage["prompt_tokens"], usage["completion_tokens"])
        episode = uuid.uuid4().hex
        self.store.put(
            f"episode/{principal}/{episode}",
            {
                "release": selection.release,
                "deployment": deployment.id,
                "generation_id": completion.get("id"),
                "usage": usage,
                "cost_usd": cost,
                "cost_known": reported_cost is not None,
                "cost_source": "provider" if reported_cost is not None else "estimate",
                "latency_ms": latency_ms,
                "created_at": time.time(),
            },
            86400,
        )
        from .adoption import Canary

        canary = Canary(self.releases)
        pointer = self.store.get("active") or {}
        rollout = pointer.get("canary")
        if rollout and reported_cost is None:
            self.store.put(
                "canary-charge/" + f"episode/{principal}/{episode}",
                {
                    "release": selection.release,
                    "deployment": deployment.id,
                    "generation_id": completion.get("id"),
                },
            )
        if rollout and deployment.id in {
            rollout["report"]["candidate"],
            rollout["report"]["incumbent"],
        }:
            canary.observe(
                selection.release,
                reported_cost,
                latency_ms=latency_ms,
                observation=f"episode/{principal}/{episode}",
            )
            canary.tick()
        self.audit.emit(
            {
                "type": "inference_completed",
                "principal": principal,
                "episode_id": episode,
                "release": selection.release,
                "deployment": deployment.id,
                "cost_known": reported_cost is not None,
            }
        )
        return {
            "episode_id": episode,
            "release": selection.release,
            "deployment": deployment.id,
            "actual_model": completion.get("model"),
            "cost_usd": cost,
            "cost_known": reported_cost is not None,
            "latency_ms": latency_ms,
        }

    def authorize(self, deployment, request, selection):
        scope = self.scope_for(selection.principal)
        if ("deployments" in scope and deployment.id not in scope["deployments"]) or (
            "models" in scope and deployment.model not in scope["models"]
        ):
            raise GatewayError(
                "unauthorized_routing", "Deployment is outside credential scope", 403
            )
        if "providers" in scope and deployment.provider not in scope["providers"]:
            raise GatewayError("unauthorized_routing", "Provider is outside credential scope", 403)
        current = EndpointPolicy.model_validate(self.releases.active()[1]["policy"])
        if not eligible(deployment, request, selection.policy) or not eligible(
            deployment, request, current
        ):
            raise GatewayError(
                "deployment_ineligible",
                "Deployment no longer satisfies current endpoint policy and eligibility",
                422,
            )

    def failure(self, selection, deployment, exc):
        from .adoption import Canary

        unsafe = isinstance(exc, GatewayError) and exc.code in {
            "upstream_protocol_error",
            "deployment_ineligible",
            "budget_overrun",
            "unauthorized_routing",
        }
        self.store.put(
            "attempt/" + uuid.uuid4().hex,
            {
                "release": selection.release,
                "deployment": deployment.id,
                "status": "failed",
                "code": getattr(exc, "code", "upstream_disconnected"),
                "cost_usd": None,
                "created_at": time.time(),
            },
        )
        canary = Canary(self.releases)
        canary.observe(selection.release, None, success=False, unsafe=unsafe)
        canary.tick()

    async def complete(self, principal, request):
        selection = self.select(principal, request)
        for deployment in selection.deployments:
            self.authorize(deployment, request, selection)
            started = time.monotonic()
            reservation = self.reserve_attempt(principal, deployment, request)
            try:
                result = await self.transport.complete(deployment, request)
                self.settle_attempt(reservation, result, deployment)
                result["x_theriad"] = self.record(
                    principal, selection, deployment, result, (time.monotonic() - started) * 1000
                )
                return result
            except GatewayError as exc:
                self.failure(selection, deployment, exc)
                if exc.code not in {"upstream_error", "provider_credentials_missing"}:
                    raise
        raise GatewayError("providers_unavailable", "All eligible deployments are unavailable", 502)

    async def stream(self, principal, request):
        selection = self.select(principal, request)
        for deployment in selection.deployments:
            emitted = False
            text, calls, usage, finish, generation = "", {}, None, None, None
            started = time.monotonic()
            reservation = None
            try:
                self.authorize(deployment, request, selection)
                reservation = self.reserve_attempt(principal, deployment, request)
                async with aclosing(
                    bounded(self.transport.stream(deployment, request))
                ) as upstream:
                    async for chunk in upstream:
                        emitted = True
                        generation = chunk.get("id") or generation
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta", {})
                            text += delta.get("content") or ""
                            finish = choice.get("finish_reason") or finish
                            for part in delta.get("tool_calls") or []:
                                index = part["index"]
                                current = calls.setdefault(
                                    index,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                current["id"] += part.get("id") or ""
                                if part.get("extra_content") is not None:
                                    from .provider_context import merge_extra_content

                                    merge_extra_content(
                                        current.setdefault("extra_content", {}),
                                        part["extra_content"],
                                    )
                                for field in ("name", "arguments"):
                                    current["function"][field] += (
                                        part.get("function", {}).get(field) or ""
                                    )
                        usage = chunk.get("usage") or usage
                        yield chunk
                self.transport.validate_message(
                    {"content": text or None, "tool_calls": list(calls.values())}
                )
                if finish is None:
                    raise GatewayError(
                        "upstream_protocol_error", "Stream ended without a finish reason", 502
                    )
                completion = {
                    "id": generation,
                    "model": deployment.model,
                    "choices": [
                        {
                            "message": {
                                "content": text or None,
                                "tool_calls": list(calls.values()),
                            },
                            "finish_reason": finish,
                        }
                    ],
                    "usage": usage,
                }
                self.settle_attempt(reservation, completion, deployment)
                metadata = self.record(
                    principal,
                    selection,
                    deployment,
                    completion,
                    (time.monotonic() - started) * 1000,
                )
                yield {"choices": [], "x_theriad": metadata, "usage": usage}
                return
            except (GatewayError, httpx.HTTPError, TimeoutError) as exc:
                self.failure(selection, deployment, exc)
                if emitted:
                    if isinstance(exc, GatewayError):
                        raise
                    raise GatewayError(
                        "upstream_disconnected", "Upstream connection failed after output", 502
                    ) from exc
                if isinstance(exc, GatewayError) and exc.code not in {
                    "upstream_error",
                    "provider_credentials_missing",
                }:
                    raise
        raise GatewayError("providers_unavailable", "All eligible deployments are unavailable", 502)

    async def native(self, principal, request):
        selection = self.select(principal, request)
        for deployment in selection.deployments:
            emitted, terminal = False, False
            started = time.monotonic()
            try:
                self.authorize(deployment, request, selection)
                reservation = self.reserve_attempt(principal, deployment, request)
                async with aclosing(
                    bounded(self.transport.responses(deployment, request))
                ) as upstream:
                    async for event in upstream:
                        result = event.get("response") if request.stream else event
                        if result and result.get("status") in {"completed", "incomplete"}:
                            usage = result.get("usage") or {}
                            calls = [
                                {
                                    "id": item["call_id"],
                                    "function": {
                                        "name": item["name"],
                                        "arguments": item.get("arguments", ""),
                                    },
                                }
                                for item in result["output"]
                                if item["type"] in {"function_call", "custom_tool_call"}
                            ]
                            completion = {
                                "id": result.get("id"),
                                "model": result.get("model"),
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
                                "choices": [{"message": {"tool_calls": calls}}],
                            }
                            self.settle_attempt(reservation, completion, deployment)
                            result["x_theriad"] = self.record(
                                principal,
                                selection,
                                deployment,
                                completion,
                                (time.monotonic() - started) * 1000,
                            )
                            for item in result["output"]:
                                if item.get("type") == "reasoning" and item.get(
                                    "encrypted_content"
                                ):
                                    self.store.put(
                                        "opaque/"
                                        + principal
                                        + "/"
                                        + hashlib.sha256(
                                            item["encrypted_content"].encode()
                                        ).hexdigest(),
                                        {"release": selection.release, "deployment": deployment.id},
                                        86400,
                                    )
                            terminal = True
                        emitted = True
                        yield event
                if not terminal:
                    raise GatewayError(
                        "upstream_protocol_error",
                        "Native stream ended without a terminal response",
                        502,
                    )
                return
            except (GatewayError, httpx.HTTPError, ValueError, TimeoutError) as exc:
                self.failure(selection, deployment, exc)
                if (
                    emitted
                    or isinstance(exc, GatewayError)
                    and exc.code not in {"upstream_error", "provider_credentials_missing"}
                ):
                    if isinstance(exc, GatewayError):
                        raise
                    raise GatewayError(
                        "upstream_disconnected", "Native response failed", 502
                    ) from exc
        raise GatewayError(
            "providers_unavailable", "All native Responses deployments unavailable", 502
        )


async def bounded(source, seconds=600):
    deadline = time.monotonic() + seconds
    async with aclosing(source):
        while True:
            try:
                event = await asyncio.wait_for(
                    anext(source), timeout=max(0.001, deadline - time.monotonic())
                )
            except StopAsyncIteration:
                return
            yield event
