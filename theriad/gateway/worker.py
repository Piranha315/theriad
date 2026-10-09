"""One elected updater: immutable discovery, bounded measurement, fitting, canary."""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone

import httpx

from .adoption import Canary, Lease, MonthlyBudget, qualify
from .contracts import (
    ChatRequest,
    Deployment,
    EndpointPolicy,
    GatewayError,
    ResponsesRequest,
    context_size,
    eligible,
)
from .releases import digest


def evaluator_identity():
    """Bind reusable measurements to the executable evaluation/protocol code."""
    import hashlib
    from pathlib import Path

    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "worker.py",
            "evaluation.py",
            "evaluation_stream.py",
            "transport.py",
            "contracts.py",
            "native.py",
            "responses.py",
            "engine.py",
            "app.py",
            "adoption.py",
        )
    }


async def fetch_catalogs(client=None, model_ids=()):
    async def fetch(client):
        response = await client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        catalogs = {"openrouter": response.json(), "fetched_at": time.time()}
        catalogs["deployment_endpoints"] = {}
        from urllib.parse import quote

        for model_id in sorted(set(model_ids)):
            endpoint_response = await client.get(
                "https://openrouter.ai/api/v1/models/" + quote(model_id, safe="/") + "/endpoints"
            )
            if endpoint_response.status_code == 200:
                catalogs["deployment_endpoints"][model_id] = endpoint_response.json().get(
                    "data", {}
                )
        key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
        if key:
            import re

            try:
                documentation = await client.get("https://artificialanalysis.ai/data-api/docs")
                match = (
                    re.search(r"Current version:.*?v(\d+\.\d+\.\d+)", documentation.text, re.S)
                    if documentation.status_code == 200
                    else None
                )
                catalogs["benchmark_methodology"] = {
                    "version": match.group(1) if match else None,
                    "source": "https://artificialanalysis.ai/data-api/docs",
                }
            except httpx.HTTPError:
                catalogs["benchmark_methodology"] = {"version": None, "state": "unavailable"}
        if key:
            combined = None
            for page in range(1, 101):
                response = await client.get(
                    "https://artificialanalysis.ai/api/v2/language/models/free",
                    headers={"x-api-key": key},
                    params={"page": page},
                )
                if response.status_code != 200:
                    catalogs["artificial_analysis"] = {
                        "unavailable": "http_" + str(response.status_code)
                    }
                    break
                data = response.json()
                if combined is None:
                    combined = {**data, "data": []}
                if data.get("intelligence_index_version") != combined.get(
                    "intelligence_index_version"
                ):
                    raise GatewayError(
                        "benchmark_version_changed", "Benchmark changed during pagination", 409
                    )
                combined["data"].extend(data.get("data", []))
                if not data.get("pagination", {}).get("has_more"):
                    combined["attribution"] = "Artificial Analysis — https://artificialanalysis.ai/"
                    combined["source"] = "https://artificialanalysis.ai/data-api/docs"
                    catalogs["artificial_analysis"] = combined
                    break
            else:
                catalogs["artificial_analysis"] = {"unavailable": "pagination_limit"}
        else:
            catalogs["artificial_analysis"] = {"unavailable": "api_key_missing"}
        return catalogs

    if client is not None:
        return await fetch(client)
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        return await fetch(client)


def discover(catalogs, mappings, policy):
    """Only explicit, exact published identities qualify; never guess by display name."""
    aa = catalogs.get("artificial_analysis", {})
    benchmark_version = aa.get("intelligence_index_version")
    models = {row["id"]: row for row in catalogs.get("openrouter", {}).get("data", [])}
    benchmarks = {row["slug"]: row for row in aa.get("data", [])}
    proposals = []
    matched = set()
    for mapping in mappings:
        model = models.get(mapping["provider_id"])
        benchmark = benchmarks.get(mapping["aa_slug"])
        canonical = (model or {}).get("canonical_slug")
        verified = (
            model is not None
            and benchmark is not None
            and canonical == mapping["revision"]
            and benchmark_version is not None
            and mapping.get("serving_variant")
            and mapping.get("reasoning_profile")
            and mapping.get("aa_id") == (benchmark or {}).get("id")
            and mapping.get("benchmark_name") == (benchmark or {}).get("name")
            and mapping.get("benchmark_version") == benchmark_version
            and mapping.get("hosting_provider")
        )
        if not verified:
            proposals.append({"state": "pending_identity", "identity": mapping})
            continue
        matched.add(model["id"])
        parameters = model.get("supported_parameters", [])
        pricing = model.get("pricing", {})
        endpoints = (
            catalogs.get("deployment_endpoints", {}).get(model["id"], {}).get("endpoints", [])
        )
        endpoint = next((e for e in endpoints if e.get("tag") == mapping["hosting_provider"]), None)
        if (
            not endpoint
            or endpoint.get("quantization") != mapping.get("serving_variant")
            or endpoint.get("quantization") in {None, "unknown"}
        ):
            proposals.append({"state": "pending_serving_variant", "identity": mapping})
            continue
        # This catalog does not attest physical serving region. A tag/display
        # name is not sufficient evidence for a jurisdictional network boundary.
        if mapping.get("region", "provider-managed") != "provider-managed":
            proposals.append({"state": "pending_region_provenance", "identity": mapping})
            continue
        parameters = endpoint.get("supported_parameters", [])
        pricing = endpoint.get("pricing", {})
        if mapping["reasoning_profile"] != "default" and (
            "reasoning" not in parameters
            or mapping.get("reasoning_config", {}).get("effort") != mapping["reasoning_profile"]
        ):
            proposals.append({"state": "pending_reasoning_configuration", "identity": mapping})
            continue
        try:
            deployment = Deployment(
                id="openrouter-" + digest(mapping)[:24],
                model=model["id"],
                revision=canonical,
                provider="openrouter",
                endpoint="https://openrouter.ai/api/v1",
                region=mapping.get("region", "provider-managed"),
                tiers=mapping["tiers"],
                context_window=endpoint["context_length"],
                max_output_tokens=endpoint["max_completion_tokens"],
                max_input_tokens=endpoint.get("max_prompt_tokens"),
                tools={"tools", "tool_choice"}.issubset(parameters),
                vision=False,
                schemas="structured_outputs" in parameters,
                reasoning="reasoning" in parameters,
                native_responses=mapping.get("native_responses") is True,
                input_price=float(pricing["prompt"]) * 1e6,
                output_price=float(pricing["completion"]) * 1e6,
                request_price=float(pricing.get("request", 0)),
                priced_at=catalogs["fetched_at"],
                credential_env="OPENROUTER_API_KEY",
                serving_variant=mapping["serving_variant"],
                reasoning_profile=mapping["reasoning_profile"],
                reasoning_config=mapping.get("reasoning_config", {}),
                hosting_provider=mapping["hosting_provider"],
                supported_parameters=parameters,
                origin=mapping.get("origin"),
            )
        except (KeyError, ValueError, TypeError):
            proposals.append({"state": "pending_metadata", "identity": mapping})
            continue
        state = "eligible" if policy.allows(deployment) else "pending_policy"
        proposals.append(
            {
                "state": state,
                "deployment": deployment.model_dump(),
                "benchmark": benchmark,
                "benchmark_version": benchmark_version,
                "mapping_hash": digest(mapping),
            }
        )
    # Expose all unpaired entries; metadata discovery does not silently omit new models.
    for model_id in sorted(set(models) - matched):
        proposals.append({"state": "pending_identity", "provider_id": model_id})
    return proposals


def grade(result, task):
    choice = result["choices"][0]
    message = choice["message"]
    if choice.get("finish_reason") in {"length", "content_filter"}:
        return 0.0, False
    kind = task["grader"]
    if kind == "protocol":
        return 1.0, True
    if kind == "exact":
        passed = (message.get("content") or "").strip() == task["expected"]
    elif kind == "json":
        try:
            passed = json.loads(message.get("content") or "") == task["expected"]
        except ValueError:
            passed = False
    elif kind == "tool_calls":
        if task.get("compatibility_case") == "tool_only" and message.get("content"):
            return 0.0, False
        try:
            calls = [
                {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                for c in message.get("tool_calls") or []
            ]
        except (ValueError, KeyError, TypeError):
            return 0.0, False
        passed = calls == task["expected"]
    else:
        raise ValueError("unsupported deterministic grader")
    return float(passed), bool(passed) if kind == "tool_calls" else True


def validate_suite(suite):
    if not suite.get("version") or not suite.get("source_revision") or not suite.get("tasks"):
        raise ValueError("versioned source-pinned evaluation suite is required")
    groups = {}
    prompt_groups = {}
    identities = set()
    for task in suite["tasks"]:
        if task["id"] in identities:
            raise ValueError("duplicate task ID")
        identities.add(task["id"])
        if task["split"] not in {"train", "validation", "qualification"}:
            raise ValueError("invalid split")
        if task["grader"] not in {"exact", "json", "tool_calls", "fixture"}:
            raise ValueError("unsupported grader")
        groups.setdefault(task["group"], set()).add(task["split"])
        for lineage in task.get("lineage_ids", []):
            groups.setdefault("lineage:" + lineage, set()).add(task["split"])
        import re

        normalized = re.sub(
            r"\s+",
            " ",
            json.dumps(
                {k: task["request"].get(k) for k in ("messages", "input", "instructions", "tools")},
                sort_keys=True,
            ),
        )
        existing_group = prompt_groups.setdefault(normalized, task["group"])
        if existing_group != task["group"]:
            raise ValueError("identical router inputs must remain in one independent task group")
        if task.get("wire_api") == "responses":
            ResponsesRequest.model_validate(task["request"])
        else:
            ChatRequest.model_validate(task["request"])
        case = task.get("compatibility_case")
        if case in {"streaming", "stream_arguments"} and not task["request"].get("stream"):
            raise ValueError("Streaming compatibility checks require upstream streaming")
        if case == "native_responses" and task.get("wire_api") != "responses":
            raise ValueError("Native Responses compatibility requires the Responses wire")
        if (
            case in {"tool_only", "parallel_tools", "stream_arguments"}
            and task["grader"] != "tool_calls"
        ):
            raise ValueError("Tool compatibility requires an exact tool-call oracle")
        if case == "parallel_tools" and len(task.get("expected", [])) < 2:
            raise ValueError("Parallel compatibility requires at least two calls")
        if case == "nullable_history" and not any(
            m.get("role") == "assistant" and m.get("content") is None and m.get("tool_calls")
            for m in task["request"].get("messages", [])
        ):
            raise ValueError("Nullable compatibility requires a tool-only assistant history")
        if case == "json_schema" and not (
            task["request"].get("response_format") or task["request"].get("text", {}).get("format")
        ):
            raise ValueError("Schema compatibility requires a structured output schema")
    if any(len(s) > 1 for s in groups.values()):
        raise ValueError("evaluation suite leaks groups between splits")


def source_provenance(suite, task):
    from theriad.routing.study import APPROVED_LICENSE_STATUSES

    source = {**suite.get("provenance", {}), **task.get("provenance", {})}
    source["source_revision"] = task.get("source_revision", suite["source_revision"])
    if source.get("license_status") not in APPROVED_LICENSE_STATUSES or any(
        not source.get(k) for k in ("license_name", "provenance_url", "collection_vintage")
    ):
        raise GatewayError(
            "provenance_pending",
            "Evaluation source requires approved license, provenance URL and collection vintage",
            409,
        )
    return source


def study_row(task, measurements):
    from theriad.routing.v2_training import ModelOutcome, StudyRow

    source = task["provenance"]
    return StudyRow(
        prompt="\n".join(str(m.get("content") or "") for m in task["request"]["messages"]),
        prompt_id=task["id"],
        split=task["split"],
        source=task.get("source", "gateway-suite"),
        benchmark_family=task["family"],
        domain=task["family"],
        template_family=task["group"],
        leakage_group=task["group"],
        source_revision=source["source_revision"],
        collection_vintage=source["collection_vintage"],
        license_name=source["license_name"],
        license_status=source["license_status"],
        provenance_url=source["provenance_url"],
        task_format="agent_state"
        if task["grader"] == "fixture"
        else "tool_use"
        if task["grader"] == "tool_calls"
        else "structured_output"
        if task["grader"] == "json"
        else "conversational_reasoning",
        collected_at=datetime.now(timezone.utc).isoformat(),
        lane="code" if task["request"].get("tools") or task["grader"] == "fixture" else "general",
        outcomes={
            model: ModelOutcome(
                success=m["success"],
                cost_usd=m["cost"],
                output_tokens=m["output_tokens"],
                latency_ms=m["latency"],
            )
            for model, m in measurements.items()
        },
    )


class UpdateWorker:
    def __init__(self, releases, transport, accounting_store=None):
        self.releases, self.store, self.transport = releases, releases.store, transport
        self.budget = MonthlyBudget(accounting_store or self.store)
        self.lease = Lease(accounting_store or self.store)

    async def measure(self, deployment, task, protocol, policy):
        if task["grader"] == "fixture":
            from .evaluation import run_fixture

            key = "fixture/result/" + digest(
                {
                    "deployment": deployment.model_dump(exclude={"priced_at"}),
                    "task": task,
                    "protocol": protocol,
                }
            )
            cached = self.store.get(key)
            if cached:
                return cached
            result = await run_fixture(self, deployment, task, protocol, policy)
            self.store.put(key, result)
            return result
        key = "measurement/" + digest(
            {
                "deployment": deployment.model_dump(exclude={"priced_at"}),
                "task": task,
                "protocol": protocol,
            }
        )
        cached = self.store.get(key)
        if cached and cached["state"] == "completed":
            return cached
        if cached:
            result = cached.get("result")
            cost = (
                await self.transport.charge(deployment, (result or {}).get("id"))
                if result
                else None
            )
            if cost is None:
                raise GatewayError(
                    "charge_unreconciled",
                    "Existing attempt requires provider charge reconciliation",
                    409,
                )
            self.budget.settle(*cached["reservation"], cost)
            success, critical = grade(result, task)
            measurement = {
                "state": "completed",
                "success": success,
                "critical_pass": critical,
                "cost": cost,
                "latency": cached["latency"],
                "output_tokens": (result.get("usage") or {}).get("completion_tokens", 0),
                "reservation": cached["reservation"],
                "protocol": protocol,
                "charge_source": "provider-generation",
                "result": result,
            }
            self.store.put(key, measurement)
            return measurement
        if task.get("wire_api") == "responses":
            from .native import NativeCodec

            request = NativeCodec(self.releases.key.decode(), self.store).prepare(
                task["request"], "evaluation:" + protocol
            )
        else:
            request = ChatRequest.model_validate(task["request"])
        if not eligible(deployment, request, policy):
            raise GatewayError(
                "candidate_ineligible",
                "Evaluation deployment fails policy/capability/context checks",
                409,
            )
        self.lease.acquire()
        reservation = self.budget.reserve(
            deployment.maximum_cost(context_size(request), request.output_limit),
            "qualification" if task["split"] == "qualification" else "pilot",
            key,
        )
        started = time.monotonic()
        # An interrupted/ambiguous call retains its full reservation and will not be reissued.
        self.store.put(key, {"state": "reserved", "reservation": reservation})
        try:
            if request._native_body is not None:
                from .evaluation_stream import collect_native

                result = await collect_native(self.transport, deployment, request)
            elif request.stream:
                from .evaluation_stream import collect_stream

                result = await collect_stream(self.transport, deployment, request)
            else:
                result = await self.transport.complete(deployment, request)
            self.store.put(
                key,
                {
                    "state": "reserved",
                    "reservation": reservation,
                    "result": result,
                    "latency": (time.monotonic() - started) * 1000,
                },
            )
            usage = result.get("usage") or {}
            cost = usage.get("cost")
            self.budget.defer(reservation, deployment, result.get("id"))
            if cost is None:
                raise GatewayError(
                    "charge_unreconciled",
                    "Provider did not report the charge; reservation retained",
                    409,
                )
            self.budget.settle(*reservation, cost)
            success, critical = grade(result, task)
            measurement = {
                "state": "completed",
                "success": success,
                "critical_pass": critical,
                "cost": cost,
                "latency": (time.monotonic() - started) * 1000,
                "output_tokens": usage.get("completion_tokens", 0),
                "reservation": reservation,
                "protocol": protocol,
                "result": result,
            }
            self.store.put(key, measurement)
            return measurement
        except GatewayError:
            raise
        except Exception as exc:
            raise GatewayError(
                "measurement_failed", "Measurement failed; reservation retained", 409
            ) from exc

    async def evaluate(self, candidate, incumbent, suite, policy, comparisons, baseline=None):
        validate_suite(suite)
        from .evaluation import required_compatibility

        cases = {
            task.get("compatibility_case")
            for task in suite["tasks"]
            if task.get("critical") is True
        }
        required = required_compatibility(candidate) | required_compatibility(incumbent)
        if required - cases:
            raise GatewayError(
                "compatibility_pending",
                "Suite lacks critical checks: " + ", ".join(sorted(required - cases)),
                409,
            )
        review = suite.get("independence_review", {})
        qualification_groups = {
            t["group"]
            for t in suite["tasks"]
            if t["split"] == "qualification" and not t.get("critical")
        }
        if (
            not review.get("reviewer")
            or review.get("source_revision") != suite["source_revision"]
            or not qualification_groups.issubset(set(review.get("reviewed_groups", [])))
        ):
            raise GatewayError(
                "independence_pending",
                "Held-out group independence requires source-pinned review",
                409,
            )
        protocol = digest(
            {
                "suite": suite,
                "candidate": candidate.model_dump(exclude={"priced_at"}),
                "incumbent": incumbent.model_dump(exclude={"priced_at"}),
                "gate_version": 1,
                "evaluator": evaluator_identity(),
            }
        )
        reviewed_tasks = [
            {**task, "provenance": source_provenance(suite, task)} for task in suite["tasks"]
        ]
        from .engine import GatewayEngine

        selector = GatewayEngine(self.releases, self.transport) if baseline is not None else None
        rows = []
        for task in reviewed_tasks:
            selected_incumbent = True
            if selector and not task.get("critical"):
                from .evaluation import TOOLS

                projection = ChatRequest.model_validate(
                    {
                        **task["request"],
                        "model": "theriad/agent",
                        **({"tools": TOOLS} if task["grader"] == "fixture" else {}),
                    }
                )
                selected_incumbent = (
                    selector.select("qualification-stratum", projection).deployments[0].id
                    == incumbent.id
                )
            a = await self.measure(incumbent, task, protocol, policy)
            b = await self.measure(candidate, task, protocol, policy)
            if a["state"] != "completed" or b["state"] != "completed":
                raise GatewayError(
                    "charge_unreconciled", "An existing attempt needs charge reconciliation", 409
                )
            rows.append(
                {
                    "group": task["group"],
                    "split": task["split"],
                    "family": task["family"],
                    "protocol": protocol,
                    "incumbent_quality": a["success"],
                    "candidate_quality": b["success"],
                    "incumbent_cost": a["cost"],
                    "candidate_cost": b["cost"],
                    "incumbent_latency": a["latency"],
                    "candidate_latency": b["latency"],
                    "critical_pass": a["critical_pass"] and b["critical_pass"],
                    "task": task,
                    "incumbent_selected": selected_incumbent,
                    "measurements": {incumbent.id: a, candidate.id: b},
                }
            )
        critical_passed = all(
            r["candidate_quality"] == 1 and r["incumbent_quality"] == 1 and r["critical_pass"]
            for r in rows
            if r["task"].get("critical")
        )
        families = sorted({r["family"] for r in rows if not r["task"].get("critical")})
        qualification_rows = [
            r
            for r in rows
            if not r["task"].get("critical")
            and (r["split"] != "qualification" or r["incumbent_selected"])
        ]
        report = qualify(qualification_rows, families, comparisons, protocol)
        if not critical_passed:
            report["qualified"] = False
            report["reasons"].append("critical_compatibility_failure")
        report["compatibility_cases"] = sorted(required)
        report["qualified_lanes"] = sorted(
            {
                "code"
                if r["task"]["request"].get("tools") or r["task"]["grader"] == "fixture"
                else "general"
                for r in qualification_rows
                if r["split"] == "qualification"
            }
        )
        report["routing_stratum"] = (
            "incumbent-selected under frozen baseline"
            if baseline is not None
            else "paired deployment"
        )
        report.update(
            protocol=protocol,
            incumbent=incumbent.id,
            candidate=candidate.id,
            comparisons=comparisons,
            identities={"candidate": candidate.model_dump(), "incumbent": incumbent.model_dump()},
            evaluation_cost_usd=sum(r["incumbent_cost"] + r["candidate_cost"] for r in rows),
        )
        return rows, report

    def fit(self, rows, candidate, incumbent, baseline):
        from theriad.models.tiers import ModelCandidate
        from theriad.routing.v2 import RouterV2Artifact
        from theriad.routing.v2_training import train_surface_artifact

        if not baseline.get("router"):
            raise GatewayError(
                "calibration_pending",
                "Baseline router artifact is required before automated replacement",
                409,
            )
        study = []
        for row in rows:
            task = row["task"]
            # Qualification remains excluded from fitting and threshold selection.
            if task["split"] == "qualification" or task.get("critical"):
                continue
            study.append(study_row(task, row["measurements"]))
        candidates = [
            ModelCandidate(
                tier=d.tiers[0],
                provider=d.provider,
                model=d.id,
                input_cost_per_million=d.input_price,
                output_cost_per_million=d.output_price,
                context_window=d.context_window,
            )
            for d in (candidate, incumbent)
        ]
        try:
            artifact = train_surface_artifact(
                study,
                fallback_models={"general": incumbent.id, "code": incumbent.id},
                active_models_by_lane={
                    "general": [incumbent.id, candidate.id],
                    "code": [incumbent.id, candidate.id],
                },
                model_candidates=candidates,
                allow_fixed_models=False,
            )
        except (ValueError, KeyError) as exc:
            raise GatewayError(
                "calibration_pending",
                "Measured training/validation data does not support a promotable fitted head",
                409,
            ) from exc
        if artifact.get("research_only"):
            raise GatewayError(
                "calibration_pending", "Fitted artifact has not passed promotion gates", 409
            )
        # Each fitted head keeps its own fitted feature scaling. Mixing heads
        # into a predecessor's normalization would silently invalidate calibration.
        artifact["router_version"] = "gateway-" + digest(artifact)[:20]
        if candidate.id not in artifact["models"]:
            raise GatewayError(
                "calibration_pending", "Insufficient fitting data for candidate", 409
            )
        RouterV2Artifact.from_dict(artifact)
        return artifact

    async def run(self, suite=None, mappings=None):
        if os.environ.get("THERIAD_GATEWAY_DISCONNECTED", "false").lower() in {"1", "true", "yes"}:
            raise GatewayError("disconnected", "External metadata refresh is disabled", 409)
        self.lease.acquire()
        report = {"started_at": time.time(), "state": "discovery", "candidates": []}
        try:
            await self.budget.reconcile(self.transport)
            await Canary(self.releases).reconcile(self.transport)
            pointer, baseline = self.releases.active()
            model_ids = [m["provider_id"] for m in mappings or []] + [
                d["model"]
                for d in baseline["deployments"]
                if d["provider"] == "openrouter" and d.get("hosting_provider")
            ]
            catalogs = await fetch_catalogs(model_ids=model_ids)
            self.store.put("snapshot/" + digest(catalogs), catalogs)
            pointer, baseline = self.releases.active()
            pointer, baseline = self.refresh_prices(catalogs, pointer, baseline)
            policy = EndpointPolicy.model_validate(baseline["policy"])
            proposals = discover(catalogs, mappings or [], policy)
            if not catalogs.get("benchmark_methodology", {}).get("version"):
                for proposal in proposals:
                    if proposal["state"] == "eligible":
                        proposal["state"] = "pending_benchmark_methodology"
            self.store.put(
                "discovery/latest", {"snapshot": digest(catalogs), "proposals": proposals}
            )
            Canary(self.releases).tick()
            if self.store.get("updates/paused") or not suite or pointer.get("canary"):
                report["state"] = (
                    "paused"
                    if self.store.get("updates/paused")
                    else "pending_configuration_or_canary"
                )
                return report
            eligible_proposals = [p for p in proposals if p["state"] == "eligible"]
            eligible_proposals.sort(
                key=lambda p: (
                    -float(
                        p["benchmark"]
                        .get("evaluations", {})
                        .get("artificial_analysis_intelligence_index")
                        or 0
                    )
                )
            )
            for proposal in eligible_proposals:
                candidate = Deployment.model_validate(proposal["deployment"])
                incumbents = [
                    Deployment.model_validate(d)
                    for d in baseline["deployments"]
                    if set(d["tiers"]) & set(candidate.tiers)
                ]
                if not incumbents or any(
                    d.revision == candidate.revision
                    and d.serving_variant == candidate.serving_variant
                    and d.hosting_provider == candidate.hosting_provider
                    and d.reasoning_config == candidate.reasoning_config
                    and d.endpoint == candidate.endpoint
                    and d.region == candidate.region
                    for d in incumbents
                ):
                    continue
                incumbent = min(incumbents, key=lambda d: d.input_price + d.output_price)
                index = (
                    "artificial_analysis_coding_index"
                    if any(t["family"] == "code" for t in suite["tasks"] if not t.get("critical"))
                    else "artificial_analysis_intelligence_index"
                )
                candidate_index = proposal["benchmark"].get("evaluations", {}).get(index)
                incumbent_benchmark = next(
                    (
                        p.get("benchmark")
                        for p in eligible_proposals
                        if p["deployment"]["model"] == incumbent.model
                        and p["deployment"]["revision"] == incumbent.revision
                    ),
                    None,
                )
                incumbent_index = (incumbent_benchmark or {}).get("evaluations", {}).get(index)
                token_cost_better = (
                    candidate.input_price <= incumbent.input_price
                    and candidate.output_price <= incumbent.output_price
                    and (
                        candidate.input_price < incumbent.input_price
                        or candidate.output_price < incumbent.output_price
                    )
                )
                headline_better = (
                    candidate_index is not None
                    and incumbent_index is not None
                    and candidate_index > incumbent_index
                )
                if not token_cost_better and not headline_better:
                    report["candidates"].append(
                        {"candidate": candidate.id, "state": "pending_shortlist_evidence"}
                    )
                    continue
                try:
                    rows, evidence = await self.evaluate(
                        candidate,
                        incumbent,
                        {
                            **suite,
                            "benchmark_version": {
                                "api": proposal["benchmark_version"],
                                "methodology": catalogs["benchmark_methodology"]["version"],
                            },
                        },
                        policy,
                        max(1, len(eligible_proposals)),
                        baseline=baseline,
                    )
                    report["candidates"].append(evidence)
                    if not evidence["qualified"]:
                        continue
                    payload = copy.deepcopy(baseline)
                    payload["deployments"].append(candidate.model_dump())
                    fitted = self.fit(rows, candidate, incumbent, baseline)
                    payload.setdefault("replacements", []).append(
                        {
                            "incumbent": incumbent.id,
                            "candidate": candidate.id,
                            "lanes": sorted(
                                {
                                    "code"
                                    if r["task"]["request"].get("tools")
                                    or r["task"]["grader"] == "fixture"
                                    else "general"
                                    for r in rows
                                    if not r["task"].get("critical")
                                }
                            ),
                            "artifact": fitted,
                        }
                    )
                    payload["evidence"] = evidence
                    release = self.releases.install(payload)
                    evidence.update(
                        release_hash=release,
                        rollback_target=pointer["release"],
                        pricing_snapshot=digest(catalogs),
                    )
                    Canary(self.releases).start(release, evidence)
                    report["state"] = "canary"
                    break
                except GatewayError as exc:
                    report["candidates"].append({"candidate": candidate.id, "state": exc.code})
                    if exc.code == "budget_overrun":
                        Canary(self.releases).tick()
                    if exc.code in {
                        "evaluation_budget_exhausted",
                        "charge_unreconciled",
                        "budget_overrun",
                    }:
                        break
            return report
        finally:
            report["finished_at"] = time.time()
            report["evaluation_budget"] = self.budget.summary()
            self.store.put("update/report/" + digest(report), report)
            self.store.put("update/latest", report)
            self.lease.release()

    def refresh_prices(self, catalogs, pointer, baseline):
        models = {m["id"]: m for m in catalogs["openrouter"].get("data", [])}

        def refresh(payload):
            payload = copy.deepcopy(payload)
            changed = False
            for deployment in payload["deployments"]:
                row = models.get(deployment["model"])
                if (
                    deployment["provider"] != "openrouter"
                    or not row
                    or row.get("canonical_slug") != deployment["revision"]
                ):
                    continue
                if deployment.get("hosting_provider"):
                    endpoints = (
                        catalogs.get("deployment_endpoints", {})
                        .get(deployment["model"], {})
                        .get("endpoints", [])
                    )
                    host = next(
                        (
                            e
                            for e in endpoints
                            if e.get("tag") == deployment["hosting_provider"]
                            and e.get("quantization") == deployment["serving_variant"]
                        ),
                        None,
                    )
                    if not host:
                        continue
                    row = {**row, "pricing": host["pricing"]}
                try:
                    prices = {
                        "input_price": float(row["pricing"]["prompt"]) * 1e6,
                        "output_price": float(row["pricing"]["completion"]) * 1e6,
                        "priced_at": catalogs["fetched_at"],
                    }
                    Deployment.model_validate({**deployment, **prices})
                except (ValueError, KeyError, TypeError):
                    continue
                deployment.update(prices)
                changed = True
            if changed:
                payload["pricing_snapshot"] = digest(catalogs)
            return payload, changed

        updated, changed = refresh(baseline)
        if not changed:
            return pointer, baseline
        release = self.releases.install(updated)
        canary = copy.deepcopy(pointer.get("canary"))
        if canary:
            original = self.releases.load(canary["candidate"])
            candidate, _ = refresh(original)
            fields = ("input_price", "output_price", "request_price", "cache_price")
            tariff_changed = any(
                any(a.get(k) != b.get(k) for k in fields)
                for a, b in zip(original["deployments"], candidate["deployments"], strict=True)
            )
            if tariff_changed:
                self.store.put(
                    "report/" + canary["candidate"],
                    {
                        **canary,
                        "state": "rolled_back",
                        "reason": "pricing_changed_requalification_required",
                    },
                )
                canary = None
            else:
                canary["candidate"] = self.releases.install(candidate)
                canary["report"] = {
                    **canary["report"],
                    "release_hash": canary["candidate"],
                    "rollback_target": release,
                }

        def change(old):
            if old != pointer:
                raise GatewayError("release_conflict", "Release changed during price refresh", 409)
            result = {
                **{k: v for k, v in old.items() if k != "canary"},
                "release": release,
                "previous": old["release"],
            }
            if canary:
                result["canary"] = canary
            return result

        pointer = self.store.mutate("active", change)
        return pointer, updated

    async def calibrate(self, suite):
        """Fit the already-approved roster from actual deployment measurements.

        This initializes portable incumbent heads; it does not add successors or
        waive their later independent qualification/canary requirements.
        """
        from theriad.models.tiers import ModelCandidate
        from theriad.routing.v2_training import train_surface_artifact

        validate_suite(suite)
        self.lease.acquire()
        report = {"state": "calibration", "started_at": time.time()}
        try:
            pointer, baseline = self.releases.active()
            if pointer.get("canary"):
                raise GatewayError(
                    "canary_conflict", "Calibration cannot replace an active canary", 409
                )
            deployments = [Deployment.model_validate(d) for d in baseline["deployments"]]
            policy = EndpointPolicy.model_validate(baseline["policy"])
            protocol = digest(
                {
                    "suite": suite,
                    "deployments": [d.model_dump(exclude={"priced_at"}) for d in deployments],
                    "calibration_version": 1,
                    "evaluator": evaluator_identity(),
                }
            )
            reviewed_tasks = [
                {**task, "provenance": source_provenance(suite, task)}
                for task in suite["tasks"]
                if task["split"] != "qualification" and not task.get("critical")
            ]
            study = []
            for task in reviewed_tasks:
                if task["split"] == "qualification" or task.get("critical"):
                    continue
                if task.get("wire_api") == "responses":
                    raise GatewayError(
                        "calibration_pending",
                        "Use portable Chat projections for fitting; native wire probes remain critical tests",
                        409,
                    )
                outcomes = {}
                for deployment in deployments:
                    projection = ChatRequest.model_validate(task["request"])
                    if not eligible(deployment, projection, policy):
                        continue
                    measurement = await self.measure(deployment, task, protocol, policy)
                    outcomes[deployment.id] = measurement
                if not outcomes:
                    continue
                study.append(study_row(task, outcomes))
            candidates = [
                ModelCandidate(
                    tier=d.tiers[0],
                    provider=d.provider,
                    model=d.id,
                    input_cost_per_million=d.input_price,
                    output_cost_per_million=d.output_price,
                    context_window=d.context_window,
                )
                for d in deployments
            ]
            fallback = min(deployments, key=lambda d: d.input_price + d.output_price).id
            try:
                artifact = train_surface_artifact(
                    study,
                    fallback_models={"general": fallback, "code": fallback},
                    active_models_by_lane={
                        "general": [d.id for d in deployments],
                        "code": [d.id for d in deployments],
                    },
                    model_candidates=candidates,
                    allow_fixed_models=False,
                )
            except (ValueError, KeyError) as exc:
                raise GatewayError(
                    "calibration_pending",
                    "Measured training/validation groups are insufficient for fitted incumbent heads",
                    409,
                ) from exc
            if artifact.get("research_only") or set(artifact["models"]) != {
                d.id for d in deployments
            }:
                raise GatewayError(
                    "calibration_pending",
                    "All approved deployments need fitted, promotable incumbent heads",
                    409,
                )
            for deployment in deployments:
                artifact["models"][deployment.id]["deployment_identity"] = {
                    key: value
                    for key, value in deployment.model_dump().items()
                    if key
                    in {
                        "revision",
                        "hosting_provider",
                        "serving_variant",
                        "reasoning_profile",
                        "reasoning_config",
                        "endpoint",
                        "region",
                    }
                }
            payload = {
                **baseline,
                "router": artifact,
                "replacements": [],
                "calibration_evidence": {
                    "protocol": protocol,
                    "suite_hash": digest(suite),
                    "groups": len(study),
                },
            }
            release = self.releases.install(payload)
            self.releases.activate(release, pointer["release"])
            report.update(
                state="calibrated",
                release=release,
                prior_release=pointer["release"],
                protocol=protocol,
            )
            return report
        except GatewayError as exc:
            report.update(state=exc.code, reason=exc.message)
            return report
        finally:
            report.update(finished_at=time.time(), evaluation_budget=self.budget.summary())
            self.store.put("calibration/latest", report)
            self.lease.release()
