import concurrent.futures
import copy
import time

import pytest

from theriad.gateway.adoption import Canary, Lease, MonthlyBudget, qualify
from theriad.gateway.contracts import Deployment, EndpointPolicy, GatewayError
from theriad.gateway.releases import Releases
from theriad.gateway.store import LocalStore
from theriad.gateway.worker import UpdateWorker, discover


def test_shared_budget_is_atomic_and_unknown_stays_reserved(tmp_path):
    store = LocalStore(tmp_path / "state.sqlite")
    budget = MonthlyBudget(store)

    def reserve(_):
        try:
            return budget.reserve(1, "pilot", "candidate")
        except GatewayError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        reservations = [r for r in pool.map(reserve, range(30)) if r]
    assert len(reservations) == 15
    budget.settle(*reservations[0], None)
    with pytest.raises(GatewayError):
        budget.reserve(0.01, "pilot", "candidate")
    budget.settle(*reservations[0], 0.5)
    budget.reserve(0.5, "pilot", "candidate")
    budget.reserve(10, "qualification", "candidate")
    with pytest.raises(GatewayError):
        budget.reserve(0.01, "qualification", "candidate")


def test_overrun_pauses_and_preserves_actual(tmp_path):
    store = LocalStore(tmp_path / "state.sqlite")
    budget = MonthlyBudget(store)
    reservation = budget.reserve(1, "pilot", "candidate")
    with pytest.raises(GatewayError):
        budget.settle(*reservation, 2)
    assert store.get(reservation[0])["entries"][reservation[1]]["actual"] == 2
    assert store.get("updates/paused")
    for bad in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            budget.reserve(bad, "pilot", "candidate")


def rows(count=100):
    return [
        {
            "group": str(i),
            "split": "qualification",
            "family": "code",
            "protocol": "p",
            "incumbent_quality": 0.8,
            "candidate_quality": 0.85,
            "incumbent_cost": 0.1,
            "candidate_cost": 0.08,
            "incumbent_latency": 100,
            "candidate_latency": 90,
            "critical_pass": True,
        }
        for i in range(count)
    ]


def test_evidence_dominance_and_fail_closed():
    assert qualify(rows(), ["code"], protocol="p")["qualified"]
    assert not qualify(rows(99), ["code"], protocol="p")["qualified"]
    measurements = rows()
    measurements[0]["candidate_cost"] = None
    assert not qualify(measurements, ["code"], protocol="p")["qualified"]


def bundle():
    artifact = {
        "schema_version": 1,
        "router_version": "test",
        "feature_mean": {},
        "feature_stdev": {},
        "lanes": {
            lane: {
                "knee_threshold": 0.5,
                "fallback_model": "baseline",
                "models": ["baseline", "challenger"],
            }
            for lane in ["general", "code"]
        },
        "models": {
            m: {
                "intercept": 3.0,
                "uncertainty": 0,
                "sample_count": 1000,
                "expected_output_tokens": 100,
            }
            for m in ["baseline", "challenger"]
        },
    }
    deployment = {
        "id": "baseline",
        "model": "test/base",
        "revision": "v1",
        "provider": "compatible",
        "endpoint": "https://provider.test/v1",
        "region": "test",
        "tiers": ["cheap_code"],
        "context_window": 64000,
        "max_output_tokens": 4096,
        "tools": True,
        "input_price": 1,
        "output_price": 2,
        "priced_at": time.time(),
    }
    return {
        "router": artifact,
        "deployments": [
            deployment,
            {**deployment, "id": "challenger", "model": "test/new", "revision": "v2"},
        ],
        "policy": {"allowed_endpoints": [deployment["endpoint"]], "allowed_regions": ["test"]},
    }


def test_signed_release_atomic_pointer_and_rollback(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    first = releases.install(bundle())
    releases.activate(first)
    changed = bundle()
    changed["router"]["router_version"] = "new-version"
    second = releases.install(changed)
    releases.activate(second, first)
    with pytest.raises(GatewayError, match="changed"):
        releases.activate(first, first)
    assert releases.rollback()["release"] == first
    envelope = releases.store.get("release/" + second)
    envelope["signature"] = "invalid"
    releases.store.put("release/" + second, envelope)
    with pytest.raises(GatewayError, match="signature"):
        releases.load(second)


def test_canary_minimums_pause_immediate_rollback_and_promotion(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    baseline = bundle()
    first = releases.install(baseline)
    releases.activate(first)
    baseline["router"]["router_version"] = "challenger"
    second = releases.install(baseline)
    clock = [1000]
    canary = Canary(releases, now=lambda: clock[0])
    report = {"qualified": True, "candidate": "challenger", "incumbent": "baseline"}
    canary.start(second, report)
    for _ in range(50):
        canary.observe(second, 0.08, latency_ms=90)
        canary.observe(first, 0.1, latency_ms=100)
    assert canary.tick()["state"] == "pending"
    clock[0] += 86400
    assert canary.tick()["state"] == "promoted"
    assert releases.active()[0]["release"] == second
    assert releases.rollback()["release"] == first
    canary.start(second, report)
    canary.observe(second, None, unsafe=True)
    assert canary.tick()["state"] == "rolled_back"
    canary.start(second, report)
    releases.store.put("updates/paused", {"reason": "operator"})
    assert canary.tick()["state"] == "rolled_back"


def test_unfitted_and_research_successors_cannot_install_or_canary(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    baseline = bundle()
    baseline["router"]["research_only"] = True
    with pytest.raises(ValueError, match="research"):
        releases.install(baseline)
    baseline = bundle()
    baseline["replacements"] = [
        {
            "incumbent": "baseline",
            "candidate": "missing",
            "lanes": ["code"],
            "artifact": baseline["router"],
        }
    ]
    with pytest.raises(ValueError, match="fitted"):
        releases.install(baseline)
    baseline = bundle()
    baseline.pop("router")
    first = releases.install(baseline)
    releases.activate(first)
    with pytest.raises(GatewayError, match="calibrated"):
        Canary(releases).start(first, {"qualified": True})


def test_discovery_exact_revision_identity_and_benchmark_versions():
    policy = EndpointPolicy(
        allowed_endpoints=["https://openrouter.ai/api/v1"], allowed_regions=["provider-managed"]
    )
    catalog = {
        "fetched_at": time.time(),
        "openrouter": {
            "data": [
                {
                    "id": "lab/model",
                    "canonical_slug": "lab/model-v1",
                    "context_length": 64000,
                    "top_provider": {"max_completion_tokens": 4096},
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["tools", "tool_choice"],
                }
            ]
        },
        "artificial_analysis": {
            "intelligence_index_version": 4.3,
            "data": [{"id": "stable-id", "slug": "model", "name": "Model (high)"}],
        },
    }
    identity = {
        "provider_id": "lab/model",
        "revision": "lab/model-v1",
        "aa_slug": "model",
        "aa_id": "stable-id",
        "benchmark_name": "Model (high)",
        "benchmark_version": 4.3,
        "serving_variant": "fp16",
        "hosting_provider": "exact-host",
        "reasoning_profile": "high",
        "reasoning_config": {"effort": "high"},
        "tiers": ["cheap_code"],
    }
    catalog["deployment_endpoints"] = {
        "lab/model": {
            "endpoints": [
                {
                    "tag": "exact-host",
                    "quantization": "fp16",
                    "context_length": 64000,
                    "max_completion_tokens": 4096,
                    "supported_parameters": ["tools", "tool_choice", "reasoning"],
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
            ]
        }
    }
    assert discover(catalog, [identity], policy)[0]["state"] == "eligible"
    for field, value in [
        ("revision", "unknown"),
        ("aa_id", "different-id"),
        ("benchmark_version", 4.2),
        ("benchmark_name", "Model (low)"),
    ]:
        assert (
            discover(catalog, [{**identity, field: value}], policy)[0]["state"]
            == "pending_identity"
        )
    missing = copy.deepcopy(catalog)
    missing["deployment_endpoints"]["lab/model"]["endpoints"][0]["pricing"].pop("prompt")
    assert discover(missing, [identity], policy)[0]["state"] == "pending_metadata"


@pytest.mark.asyncio
async def test_interrupted_measurement_retains_reservation_and_does_not_reissue(tmp_path):
    class BrokenTransport:
        calls = 0

        async def complete(self, deployment, request):
            self.calls += 1
            raise GatewayError("upstream_error", "interrupted", 502)

        async def charge(self, deployment, generation):
            return None

    store = LocalStore(tmp_path / "state.sqlite")
    releases = Releases(store, "s" * 32)
    transport = BrokenTransport()
    worker = UpdateWorker(releases, transport)
    base = bundle()
    deployment = Deployment.model_validate(base["deployments"][0])
    task = {
        "id": "test",
        "group": "g",
        "split": "qualification",
        "family": "code",
        "grader": "exact",
        "expected": "ok",
        "request": {"messages": [{"role": "user", "content": "Reply ok"}]},
    }
    with pytest.raises(GatewayError):
        await worker.measure(
            deployment, task, "protocol", EndpointPolicy.model_validate(base["policy"])
        )
    with pytest.raises(GatewayError, match="reconciliation"):
        await worker.measure(
            deployment, task, "protocol", EndpointPolicy.model_validate(base["policy"])
        )
    assert transport.calls == 1
    ledger = store.get(worker.budget.key())
    assert len(ledger["entries"]) == 1
    assert next(iter(ledger["entries"].values()))["actual"] is None
    measurements = rows()
    measurements += [{**measurements[0], "split": "train"}]
    assert "split_leakage" in qualify(measurements, ["code"], protocol="p")["reasons"]
    measurements = rows()
    for row in measurements:
        row["candidate_quality"] = 0.79
    assert not qualify(measurements, ["code"])["qualified"]
    assert not qualify(rows(), ["code"], protocol="changed")["qualified"]


def test_lease_excludes_duplicate_workers(tmp_path):
    store = LocalStore(tmp_path / "state.sqlite")
    first, second = Lease(store), Lease(store)
    first.acquire()
    with pytest.raises(GatewayError):
        second.acquire()
    first.release()
    second.acquire()


def test_canary_failures_do_not_count_as_completed_candidate_requests(tmp_path):
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    first = releases.install(bundle())
    releases.activate(first)
    second = releases.install({**bundle(), "source": "candidate"})
    clock = [1000]
    canary = Canary(releases, now=lambda: clock[0])
    canary.start(second, {"qualified": True, "candidate": "challenger", "incumbent": "baseline"})
    for _ in range(49):
        canary.observe(second, 0.08, latency_ms=90)
    canary.observe(second, 0, success=False, latency_ms=90)
    for _ in range(50):
        canary.observe(first, 0.1, latency_ms=100)
    clock[0] += 86400
    assert canary.tick()["state"] == "pending"
    canary.observe(second, 0.08, latency_ms=90)
    assert canary.tick()["state"] == "promoted"
