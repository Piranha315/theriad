import concurrent.futures
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from theriad.gateway.adoption import Canary, MonthlyBudget
from theriad.gateway.contracts import ChatRequest, Deployment, GatewayError
from theriad.gateway.engine import GatewayEngine, bounded
from theriad.gateway.gcp import GoogleCredentials
from theriad.gateway.releases import Releases
from theriad.gateway.store import LocalStore
from theriad.gateway.worker import UpdateWorker, validate_suite

from .test_adoption import bundle


def module(name):
    path = Path(__file__).resolve().parents[3] / "infra" / "gateway-gcp" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_private_pilot_release_does_not_change_shared_authority(tmp_path):
    shared = LocalStore(tmp_path / "authority.sqlite")
    releases = Releases(LocalStore(tmp_path / "pilot.sqlite"), "s" * 32)
    releases.activate(releases.install(bundle()))
    engine = GatewayEngine(releases, object(), shared)
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    selected = engine.select("eval:qualification:pilot", request)
    reservation = engine.reserve_attempt(
        "eval:qualification:pilot", selected.deployments[0], request
    )
    assert shared.get(reservation[0])["entries"][reservation[1]]["phase"] == "qualification"
    assert not shared.get("active")
    assert not releases.store.get(reservation[0])
    assert engine.reserve_attempt("ordinary-inference", selected.deployments[0], request) is None


@pytest.mark.asyncio
async def test_charge_reconciliation_survives_month_boundary_and_restart(tmp_path):
    clock = [datetime(2026, 10, 1, 3, 59, tzinfo=timezone.utc).timestamp()]
    store = LocalStore(tmp_path / "budget.sqlite")
    budget = MonthlyBudget(store, now=lambda: clock[0])
    reservation = budget.reserve(0.1, "pilot", "attempt")
    assert reservation[0] == "budget/2026-09"
    deployment = Deployment.model_validate(bundle()["deployments"][0])
    budget.defer(reservation, deployment, "generation-1")
    clock[0] += 120
    assert budget.key() == "budget/2026-10"

    class Transport:
        async def charge(self, endpoint, generation):
            assert endpoint.id == "baseline" and generation == "generation-1"
            return 0.08

    restarted = MonthlyBudget(LocalStore(store.path), now=lambda: clock[0])
    await restarted.reconcile(Transport())
    assert store.get(reservation[0])["entries"][reservation[1]]["actual"] == 0.08
    assert store.get("evaluation/charges") == {}


@pytest.mark.asyncio
async def test_canary_unknown_charge_reconciliation_and_early_regression(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    first = releases.install(bundle())
    releases.activate(first)
    candidate = bundle()
    candidate["source"] = "new"
    second = releases.install(candidate)
    canary = Canary(releases)
    canary.start(second, {"qualified": True, "candidate": "challenger", "incumbent": "baseline"})
    key = "episode/alice/e1"
    releases.store.put(
        key, {"release": second, "deployment": "challenger", "generation_id": "g"}, 86400
    )
    canary.observe(second, None, observation=key, latency_ms=90)

    class Transport:
        async def charge(self, deployment, generation):
            return 0.08

    await canary.reconcile(Transport())
    record = releases.active()[0]["canary"]
    assert not record["unknown_cost"] and record["candidate_count"] == 1
    assert releases.store.get(key)["cost_known"]
    await canary.reconcile(Transport())
    assert releases.active()[0]["canary"]["candidate_cost"] == 0.08
    for _ in range(49):
        canary.observe(second, 0.2, latency_ms=90)
    for _ in range(50):
        canary.observe(first, 0.1, latency_ms=100)
    assert canary.tick()["state"] == "rolled_back"  # No need to wait 24h to detect the regression.


def test_google_refresh_is_serialized_and_not_static():
    class Credential:
        valid = False
        expired = True
        token = "old"
        refreshes = 0

        def refresh(self, request):
            self.refreshes += 1
            self.token = "new-" + str(self.refreshes)
            self.valid, self.expired = True, False

    credential = Credential()
    resolver = GoogleCredentials(credential)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(lambda _: resolver.token(), range(20))) == {"new-1"}
    credential.expired = True
    assert resolver.token() == "new-2"


def test_credit_record_requires_current_evidence_and_all_services():
    preflight = module("preflight")
    record = {
        "project_id": "project",
        "billing_account": "account",
        "remaining_usd": 100,
        "verified_at": "2026-09-30T12:00:00Z",
        "expires_at": "2026-10-30T12:00:00Z",
        "eligible_services": [
            "Cloud Run",
            "Vertex AI",
            "Cloud Storage",
            "Firestore",
            "Cloud Scheduler",
        ],
        "evidence_reference": "billing portal export",
    }
    now = datetime(2026, 9, 30, 13, tzinfo=timezone.utc)
    preflight.validate_credit(record, "project", now)
    for patch in (
        {"remaining_usd": float("nan")},
        {"remaining_usd": 0},
        {"eligible_services": []},
        {"evidence_reference": ""},
        {"verified_at": "2026-09-29T12:00:00Z"},
    ):
        with pytest.raises(ValueError):
            preflight.validate_credit(record | patch, "project", now)


def test_build_context_excludes_workspace_secrets_and_ui(tmp_path):
    context = module("build_context").build_context(tmp_path / "container")
    assert not (context / ".env").exists()
    assert not (context / "theriad-ui").exists()
    assert not (context / ".git").exists()
    assert (context / "theriad/gateway/app.py").is_file()


def test_gateway_only_deployment_does_not_require_evaluation_inputs(monkeypatch):
    import sys

    preflight = module("preflight")
    monkeypatch.setitem(sys.modules, "preflight", preflight)
    deploy = module("deploy")
    now = datetime.now(timezone.utc)
    from datetime import timedelta

    credit = {
        "project_id": "project",
        "billing_account": "account",
        "remaining_usd": 20,
        "verified_at": now.isoformat(),
        "expires_at": (now + timedelta(days=30)).isoformat(),
        "eligible_services": [
            "Cloud Run", "Vertex AI", "Cloud Storage", "Firestore", "Cloud Scheduler"
        ],
        "evidence_reference": "verified billing ledger",
    }
    inputs = {
        "project_id": "project",
        "billing_account": "account",
        "authorized_invokers": ["user:operator@example.com"],
        "image": "us-central1-docker.pkg.dev/project/pilot/gateway@sha256:" + "a" * 64,
        "bucket_name": "private-pilot-bucket",
        "release_signing_secret": "signing",
        "gateway_tokens_secret": "tokens",
        "openrouter_secret": "provider",
        "enable_updates": False,
        "scheduler_paused": True,
    }
    deploy.check_inputs(inputs, credit)
    deploy.check_inputs(inputs | {"openrouter_secret": ""}, credit)
    for patch in (
        {"enable_updates": True},
        {"image": "mutable-image:latest"},
        {"authorized_invokers": []},
        {"scheduler_paused": False},
        {"network": "network-without-subnet"},
    ):
        with pytest.raises(ValueError):
            deploy.check_inputs(inputs | patch, credit)
    worker_inputs = inputs | {
        "enable_updates": True,
        "artificial_analysis_secret": "analysis",
        "suite_object": "configuration/reviewed-suite.json",
        "identities_object": "configuration/verified-identities.json",
        "suite_sha256": "b" * 64,
        "identities_sha256": "c" * 64,
    }
    deploy.check_inputs(worker_inputs, credit)
    with pytest.raises(ValueError, match="SHA-256"):
        deploy.check_inputs(worker_inputs | {"suite_sha256": "placeholder"}, credit)


@pytest.mark.asyncio
async def test_stream_cancellation_closes_upstream():
    closed = []

    async def source():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    stream = bounded(source())
    assert await anext(stream) == "first"
    await stream.aclose()
    assert closed == [True]


def test_duplicate_router_inputs_cannot_inflate_independent_groups():
    task = {
        "id": "a",
        "group": "g1",
        "split": "train",
        "grader": "exact",
        "request": {"messages": [{"role": "user", "content": "one"}]},
    }
    suite = {
        "version": "v1",
        "source_revision": "sha",
        "tasks": [task, task | {"id": "b", "group": "g2", "split": "qualification"}],
    }
    with pytest.raises(ValueError, match="independent task group"):
        validate_suite(suite)


def test_price_change_ends_canary_and_requires_requalification(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    original = bundle()
    for d in original["deployments"]:
        d.update(provider="openrouter", endpoint="https://openrouter.ai/api/v1")
    original["policy"]["allowed_endpoints"] = ["https://openrouter.ai/api/v1"]
    first = releases.install(original)
    releases.activate(first)
    changed = dict(original, source="candidate")
    second = releases.install(changed)
    Canary(releases).start(
        second, {"qualified": True, "candidate": "challenger", "incumbent": "baseline"}
    )
    pointer, payload = releases.active()
    catalogs = {
        "fetched_at": datetime.now().timestamp(),
        "openrouter": {
            "data": [
                {
                    "id": d["model"],
                    "canonical_slug": d["revision"],
                    "pricing": {"prompt": "0.000003", "completion": "0.000004"},
                }
                for d in payload["deployments"]
            ]
        },
    }
    pointer, _ = UpdateWorker(releases, object()).refresh_prices(catalogs, pointer, payload)
    assert not pointer.get("canary")
    assert (
        releases.store.get("report/" + second)["reason"]
        == "pricing_changed_requalification_required"
    )


def test_credential_deployment_scope_is_applied_before_selection(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    releases.activate(releases.install(bundle()))
    engine = GatewayEngine(releases, object())
    engine.scopes["alice"] = {"deployments": ["challenger"]}
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    selection = engine.select("alice", request)
    assert [d.id for d in selection.deployments] == ["challenger"]
    engine.scopes["alice"] = {"deployments": ["baseline"]}
    with pytest.raises(GatewayError, match="credential scope"):
        engine.authorize(selection.deployments[0], request, selection)


@pytest.mark.asyncio
async def test_concurrent_credentials_of_same_principal_keep_separate_scopes(tmp_path):
    import asyncio

    from theriad.gateway.ports import CREDENTIAL_SCOPE

    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    releases.activate(releases.install(bundle()))
    engine = GatewayEngine(releases, object())
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])

    async def select(deployment):
        token = CREDENTIAL_SCOPE.set(("alice", {"deployments": [deployment]}))
        try:
            await asyncio.sleep(0)
            return [d.id for d in engine.select("alice", request).deployments]
        finally:
            CREDENTIAL_SCOPE.reset(token)

    assert await asyncio.gather(select("baseline"), select("challenger")) == [
        ["baseline"],
        ["challenger"],
    ]
    engine.scopes["alice"] = {"deployments": []}
    with pytest.raises(GatewayError, match="No authorized deployment"):
        engine.select("alice", request)


def test_bridge_preserves_theriad_key_separately_from_google_iam_and_query():
    import httpx
    from fastapi.testclient import TestClient

    from theriad.gateway.bridge import create_bridge

    requests = []

    class Identity:
        def token(self):
            return "google-id-token"

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"object": "list", "data": []})

    app = create_bridge("https://gateway.test", Identity(), httpx.MockTransport(handler))
    response = TestClient(app).get(
        "/v1/models?client_version=0.145.0", headers={"Authorization": "Bearer theriad-key"}
    )
    assert response.status_code == 200
    assert requests[0].headers["Authorization"] == "Bearer theriad-key"
    assert requests[0].headers["X-Serverless-Authorization"] == "Bearer google-id-token"
    assert requests[0].url.params["client_version"] == "0.145.0"


def test_native_default_output_bound_and_no_reasoning_semantics():
    from theriad.gateway.contracts import EndpointPolicy, eligible
    from theriad.gateway.transport import WireTransport

    deployment = Deployment.model_validate(bundle()["deployments"][0] | {"native_responses": True})
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        reasoning={"effort": "none", "summary": "none"},
    )
    request._native_body = {"input": "hello", "reasoning": {"effort": "none", "summary": "none"}}
    assert eligible(deployment, request, EndpointPolicy.model_validate(bundle()["policy"]))
    body = WireTransport.body(deployment, request)
    assert body["max_output_tokens"] == request.output_limit
    assert "reasoning" not in body
    request.reasoning = {"effort": "high"}
    assert not eligible(deployment, request, EndpointPolicy.model_validate(bundle()["policy"]))


def test_canary_does_not_change_explicit_pins_or_broaden_credential_scope(tmp_path, monkeypatch):
    import uuid

    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    first = releases.install(bundle())
    releases.activate(first)
    second = releases.install({**bundle(), "source": "candidate"})
    Canary(releases).start(
        second,
        {
            "qualified": True,
            "candidate": "challenger",
            "incumbent": "baseline",
            "qualified_lanes": ["general"],
        },
    )
    monkeypatch.setattr("theriad.gateway.engine.uuid.uuid4", lambda: uuid.UUID(int=0))
    engine = GatewayEngine(releases, object())
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    assert engine.select("alice", request).release == second
    request.model = "test/base"
    assert engine.select("alice", request).release == first
    request.model = "theriad/agent"
    engine.scopes["alice"] = {"deployments": ["baseline"]}
    assert engine.select("alice", request).release == first


def test_vendor_filter_cannot_be_bypassed_with_unknown_origin(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    payload = bundle()
    payload["policy"]["excluded_origins"] = ["prohibited-vendor"]
    for deployment in payload["deployments"]:
        deployment["origin"] = None
    releases.activate(releases.install(payload))
    engine = GatewayEngine(releases, object())
    with pytest.raises(GatewayError, match="No authorized deployment"):
        engine.select("alice", ChatRequest(messages=[{"role": "user", "content": "hello"}]))


def test_private_release_databases_default_to_one_local_budget_authority(tmp_path, monkeypatch):
    from theriad.gateway.ports import evaluation_store

    monkeypatch.delenv("THERIAD_EVALUATION_DB", raising=False)
    monkeypatch.delenv("THERIAD_EVALUATION_STORE", raising=False)
    expand = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda path: (
            tmp_path / "authority.sqlite"
            if str(path) == "~/.theriad/gateway/state.sqlite"
            else expand(path)
        ),
    )
    first = evaluation_store(LocalStore(tmp_path / "first-release.sqlite"))
    second = evaluation_store(LocalStore(tmp_path / "second-release.sqlite"))
    assert first.path == second.path == tmp_path / "authority.sqlite"
    MonthlyBudget(first).reserve(14, "pilot", "first")
    with pytest.raises(GatewayError, match="allowance"):
        MonthlyBudget(second).reserve(2, "pilot", "second")
