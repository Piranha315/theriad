"""Budget, evidence gates and automatic release lifecycle. No inference on import."""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .contracts import GatewayError
from .releases import digest


class MonthlyBudget:
    def __init__(self, store, now=None):
        self.store = store
        self.now = now or time.time

    def key(self):
        month = datetime.fromtimestamp(self.now(), ZoneInfo("America/New_York")).strftime("%Y-%m")
        return "budget/" + month

    def summary(self):
        entries = (self.store.get(self.key()) or {}).get("entries", {})
        known = sum(e["actual"] for e in entries.values() if e.get("actual") is not None)
        unresolved = sum(e["maximum"] for e in entries.values() if e.get("actual") is None)
        return {
            "month": self.key().split("/")[1],
            "cap_usd": 25,
            "settled_usd": known,
            "unresolved_committed_usd": unresolved,
            "available_usd": max(0, 25 - known - unresolved),
            "unresolved_attempts": sum(e.get("actual") is None for e in entries.values()),
            "allocations_usd": {"pilot": 15, "qualification": 10},
        }

    def reserve(self, maximum: float, phase: str, identity: str):
        if not math.isfinite(maximum) or maximum < 0 or phase not in {"pilot", "qualification"}:
            raise ValueError("invalid reservation")
        if self.store.get("updates/paused"):
            raise GatewayError("updates_paused", "Paid evaluation is paused", 409)
        key = self.key()
        reservation = uuid.uuid4().hex

        def change(old):
            old = old or {"entries": {}}
            entries = old["entries"]
            committed = sum(
                e["actual"] if e.get("actual") is not None else e["maximum"]
                for e in entries.values()
            )
            phase_total = sum(
                e["actual"] if e.get("actual") is not None else e["maximum"]
                for e in entries.values()
                if e["phase"] == phase
            )
            if (
                committed + maximum > 25 + 1e-10
                or phase_total + maximum > (15 if phase == "pilot" else 10) + 1e-10
            ):
                raise GatewayError(
                    "evaluation_budget_exhausted", "Monthly evaluation allowance is exhausted", 409
                )
            entries[reservation] = {
                "maximum": maximum,
                "actual": None,
                "phase": phase,
                "identity": identity,
                "created_at": self.now(),
            }
            return old

        self.store.mutate(key, change)
        return key, reservation

    def settle(self, key, reservation, actual):
        if actual is not None and (not math.isfinite(actual) or actual < 0):
            raise ValueError("invalid cost")

        def change(old):
            if not old or reservation not in old["entries"]:
                raise ValueError("reservation not found")
            entry = old["entries"][reservation]
            if entry["actual"] is not None:
                if entry["actual"] != actual:
                    raise ValueError("reservation is already settled")
                return old
            if actual is not None and actual > entry["maximum"] + 1e-10:
                # Persist the actual charge, then stop further work. Never erase spend.
                entry["overrun"] = True
            entry["actual"] = actual
            return old

        result = self.store.mutate(key, change)
        if result["entries"][reservation].get("overrun"):
            self.store.put("updates/paused", {"reason": "budget_overrun"})
            raise GatewayError(
                "budget_overrun", "Actual charge exceeded reservation; updates paused", 409
            )

    def defer(self, reservation, deployment, generation):
        if not generation:
            return
        key, identifier = reservation
        self.store.mutate(
            "evaluation/charges",
            lambda old: {
                **(old or {}),
                identifier: {
                    "ledger": key,
                    "reservation": identifier,
                    "deployment": deployment.model_dump(),
                    "generation": generation,
                },
            },
        )

    async def reconcile(self, transport):
        from .contracts import Deployment

        for identifier, record in list((self.store.get("evaluation/charges") or {}).items())[:128]:
            ledger = self.store.get(record["ledger"])
            entry = (ledger or {}).get("entries", {}).get(identifier)
            if entry and entry.get("actual") is None:
                try:
                    actual = await transport.charge(
                        Deployment.model_validate(record["deployment"]), record["generation"]
                    )
                except Exception:
                    continue
                if actual is None:
                    continue
                self.settle(record["ledger"], identifier, actual)
            self.store.mutate(
                "evaluation/charges",
                lambda old, identifier=identifier: {
                    k: v for k, v in (old or {}).items() if k != identifier
                },
            )


class Lease:
    def __init__(self, store, name="updater", now=None):
        self.store, self.key = store, "lease/" + name
        self.owner = uuid.uuid4().hex
        self.now = now or time.time

    def acquire(self, seconds=1800):
        def change(old):
            if old and old["expires"] > self.now() and old["owner"] != self.owner:
                raise GatewayError(
                    "worker_busy", "An authoritative update worker is already running", 409
                )
            return {"owner": self.owner, "expires": self.now() + seconds}

        self.store.mutate(self.key, change)

    def release(self):
        def change(old):
            return (
                {"owner": self.owner, "expires": 0} if old and old["owner"] == self.owner else old
            )

        self.store.mutate(self.key, change)


def qualify(rows, families, comparisons=1, protocol=None, bootstrap_samples=10000):
    """Paired group bootstrap with Bonferroni simultaneous confidence bounds.

    Bounds cover quality, total cost and latency for every family/challenger.
    Test data cannot share group identities with training or validation data.
    """
    result = {"qualified": False, "families": {}, "reasons": [], "evidence_hash": digest(rows)}
    if comparisons < 1 or not families:
        raise ValueError("families and comparison count are required")
    group_splits = {}
    for row in rows:
        group_splits.setdefault(row["group"], set()).add(row["split"])
        if row["split"] not in {"train", "validation", "qualification"}:
            result["reasons"].append("invalid_split")
        if protocol and row.get("protocol") != protocol:
            result["reasons"].append("protocol_mismatch")
    if any(len(splits) != 1 for splits in group_splits.values()):
        result["reasons"].append("split_leakage")
    rng = np.random.default_rng(20260929)
    alpha = 0.05 / (4 * len(families) * comparisons)
    for family in families:
        groups = {}
        for row in rows:
            if row["split"] != "qualification" or row["family"] != family:
                continue
            values = [
                row.get(k)
                for k in (
                    "incumbent_quality",
                    "candidate_quality",
                    "incumbent_cost",
                    "candidate_cost",
                    "incumbent_latency",
                    "candidate_latency",
                )
            ]
            if any(
                v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in values
            ):
                result["reasons"].append("incomplete_measurement")
                continue
            if max(values[:2]) > 1 or row.get("critical_pass") is not True:
                result["reasons"].append("critical_compatibility_failure")
            groups.setdefault(row["group"], []).append(values)
        if len(groups) < 100:
            result["reasons"].append("insufficient_groups:" + family)
            continue
        matrix = np.asarray([np.mean(values, axis=0) for values in groups.values()])
        draws = np.mean(
            matrix[rng.integers(0, len(matrix), (bootstrap_samples, len(matrix)))], axis=1
        )
        quality = draws[:, 1] - draws[:, 0]
        cost = draws[:, 3] - draws[:, 2]
        saving = draws[:, 2] * 0.95 - draws[:, 3]
        latency = draws[:, 5] - draws[:, 4] * 1.10
        bounds = {
            "groups": len(groups),
            "quality_lower": float(np.quantile(quality, alpha)),
            "cost_delta_upper": float(np.quantile(cost, 1 - alpha)),
            "saving_lower": float(np.quantile(saving, alpha)),
            "latency_delta_upper": float(np.quantile(latency, 1 - alpha)),
            "alpha": alpha,
        }
        passed = (
            bounds["quality_lower"] >= 0
            and bounds["cost_delta_upper"] <= 0
            and (
                bounds["quality_lower"] >= 0.02
                or (bounds["saving_lower"] >= 0 and float(matrix[:, 2].mean()) > 0)
            )
            and bounds["latency_delta_upper"] <= 0
        )
        bounds["passed"] = passed
        result["families"][family] = bounds
        if not passed:
            result["reasons"].append("dominance_not_established:" + family)
    result["qualified"] = not result["reasons"] and len(result["families"]) == len(families)
    return result


class Canary:
    def __init__(self, releases, now=None):
        self.releases, self.store = releases, releases.store
        self.now = now or time.time

    def start(self, release, report):
        if not report.get("qualified"):
            raise GatewayError("promotion_blocked", "Qualification failed", 409)
        payload = self.releases.load(release)
        if not payload.get("router"):
            raise GatewayError("promotion_blocked", "A calibrated router artifact is required", 409)
        deployment = next(
            (d for d in payload["deployments"] if d["id"] == report.get("candidate")), {}
        )
        record = {
            "candidate": release,
            "report": report,
            "started": self.now(),
            "fraction": 0.05,
            "candidate_count": 0,
            "control_count": 0,
            "candidate_completed": 0,
            "control_completed": 0,
            "candidate_failures": 0,
            "control_failures": 0,
            "candidate_cost": 0.0,
            "control_cost": 0.0,
            "unknown_cost": False,
            "pending_charges": {},
            "limits": {
                "cost_ratio": deployment.get("canary_cost_ratio", 1.05),
                "latency_ratio": deployment.get("canary_latency_ratio", 1.10),
                "failure_delta": deployment.get("canary_failure_delta", 0.02),
            },
            "unsafe": False,
        }

        def change(old):
            if not old or old.get("canary"):
                raise GatewayError("canary_conflict", "No baseline or an existing canary", 409)
            return {**old, "canary": record}

        self.store.mutate("active", change)
        self.store.put("report/" + release, record)

    def observe(self, release, cost, success=True, unsafe=False, latency_ms=None, observation=None):
        def change(old):
            if not old or not old.get("canary"):
                return old
            c = old["canary"]
            c["unsafe"] |= unsafe
            arm = "candidate" if release == c["candidate"] else "control"
            if release not in {c["candidate"], old["release"]}:
                return old
            c[arm + "_count"] += 1
            c[arm + "_completed"] = c.get(arm + "_completed", 0) + int(success)
            c[arm + "_failures"] += int(not success)
            if cost is None:
                c.setdefault("pending_charges", {})[
                    observation or ("unknown/" + uuid.uuid4().hex)
                ] = arm
                c["unknown_cost"] = True
            else:
                c[arm + "_cost"] += cost
            if latency_ms is not None:
                c.setdefault(arm + "_latencies", []).append(latency_ms)
                c[arm + "_latencies"] = c[arm + "_latencies"][-500:]
            c["unsafe"] |= unsafe
            return old

        self.store.mutate("active", change)

    async def reconcile(self, transport):
        from .contracts import Deployment

        pointer = self.store.get("active") or {}
        canary = pointer.get("canary")
        if not canary:
            return
        for key in list(canary.get("pending_charges", {}))[:128]:
            episode = self.store.get(key) or self.store.get("canary-charge/" + key)
            if not episode or not episode.get("generation_id"):
                continue
            payload = self.releases.load(episode["release"])
            deployment = next(
                (d for d in payload["deployments"] if d["id"] == episode["deployment"]), None
            )
            if not deployment:
                continue
            try:
                cost = await transport.charge(
                    Deployment.model_validate(deployment), episode["generation_id"]
                )
            except Exception:
                continue
            if cost is None:
                continue

            def change(old, key=key, cost=cost):
                current = (old or {}).get("canary")
                if not current or current["started"] != canary["started"]:
                    return old
                arm = current.get("pending_charges", {}).pop(key, None)
                if arm:
                    current[arm + "_cost"] += cost
                current["unknown_cost"] = bool(current.get("pending_charges"))
                return old

            self.store.mutate("active", change)
            self.store.mutate(
                key,
                lambda old, cost=cost: (
                    {
                        **old,
                        "cost_usd": cost,
                        "cost_known": True,
                        "cost_source": "provider-generation",
                    }
                    if old
                    else old
                ),
            )

    def tick(self):
        paused = self.store.get("updates/paused")
        decision = {}

        def change(old):
            if not old or not old.get("canary"):
                return old
            c = old["canary"]
            decision.update(c)
            if c["unsafe"] or paused:
                decision["state"] = "rolled_back"
                return {k: v for k, v in old.items() if k != "canary"}
            candidate, control = c["candidate_count"], c["control_count"]
            if (
                c.get("candidate_completed", candidate) < 50
                or c.get("control_completed", control) < 50
                or c["unknown_cost"]
            ):
                decision["state"] = "pending"
                return old
            # Operational canary is a reliability/cost tripwire, not new quality evidence.
            limits = c.get(
                "limits", {"cost_ratio": 1.05, "latency_ratio": 1.10, "failure_delta": 0.02}
            )
            bad = (
                c["candidate_failures"] / candidate
                > c["control_failures"] / control + limits["failure_delta"]
            )
            bad |= (
                c["candidate_cost"] / candidate > c["control_cost"] / control * limits["cost_ratio"]
            )
            a, b = c.get("candidate_latencies", []), c.get("control_latencies", [])
            if len(a) < 50 or len(b) < 50:
                decision["state"] = "pending"
                return old
            bad |= (
                float(np.quantile(a, 0.95)) > float(np.quantile(b, 0.95)) * limits["latency_ratio"]
            )
            if not bad and self.now() - c["started"] < 86400:
                decision["state"] = "pending"
                return old
            decision["state"] = "rolled_back" if bad else "promoted"
            if bad:
                return {k: v for k, v in old.items() if k != "canary"}
            return {
                "release": c["candidate"],
                "previous": old["release"],
                "activated_at": self.now(),
            }

        self.store.mutate("active", change)
        if decision:
            self.store.put("report/" + decision["candidate"], decision)
            self.store.put("report/history/" + digest(decision), decision)
        return decision
