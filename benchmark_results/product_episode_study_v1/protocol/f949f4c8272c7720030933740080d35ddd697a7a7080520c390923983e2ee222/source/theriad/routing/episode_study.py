"""Budget-bounded, paired product-episode evaluation.

This module deliberately does not infer missing counterfactuals. A locked test
compares two independently executed policies on the same resettable fixtures.
Provider execution is supplied by the real product adapters, never by the
statistical evaluator.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .study_budget import StudyBudgetConfig, StudyBudgetExceeded, StudyBudgetLedger

VERSION = "product-episodes-v1"
FAMILIES = ("chat", "coding", "tools", "search", "agents")
STAGE_CAPS = {"pilot": 10.0, "development": 30.0, "test": 50.0, "smoke": 10.0}
MIN_INFERENCE_GROUPS_PER_FAMILY = 20


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_event(path: Path, value: Mapping[str, Any]) -> None:
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # Never silently skip a damaged tail: an unaccounted call may be charged.
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("event ledger contains a non-object record")
    return records


@dataclass(frozen=True)
class Episode:
    episode_id: str
    group_id: str
    family: str
    split: str
    prompt: str
    fixture: dict[str, Any]
    grader: dict[str, Any]
    capabilities: tuple[str, ...] = ("text",)
    tags: tuple[str, ...] = ()
    fixture_version: str = VERSION
    grader_version: str = VERSION
    source: str = "authored"
    template_id: str = ""
    max_calls: int = 1
    max_output_tokens: int = 2048
    max_input_tokens: int = 24000
    max_seconds: int = 180

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Episode:
        fields = dict(row)
        for key in ("capabilities", "tags"):
            if key in fields:
                fields[key] = tuple(fields[key])
        return cls(**fields)

    @property
    def content_hash(self) -> str:
        return digest(asdict(self))


def validate_episodes(rows: Sequence[Episode], *, require_all_families: bool = True) -> dict[str, Any]:
    ids: set[str] = set()
    groups: dict[str, str] = {}
    templates: dict[str, str] = {}
    prompts: dict[str, str] = {}
    related_groups: dict[str, str] = {}
    group_families: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.episode_id in ids or not row.episode_id or not row.group_id:
            raise ValueError("duplicate episode or missing independent group identifier")
        ids.add(row.episode_id)
        if row.family not in FAMILIES or row.split not in {"pilot", "train", "validation", "test", "smoke"}:
            raise ValueError("invalid episode family/split")
        if not row.prompt.strip() or not row.grader.get("kind") or not row.fixture_version or not row.grader_version:
            raise ValueError("every episode needs a prompt, pinned fixture, and explicit grader")
        bounds = (row.max_calls, row.max_output_tokens, row.max_input_tokens, row.max_seconds)
        if any(type(bound) is not int or bound <= 0 for bound in bounds):
            raise ValueError("episode execution must be bounded")
        for registry, key in ((groups, row.group_id), (templates, row.template_id), (prompts, digest(" ".join(row.prompt.split())))):
            if not key:
                continue
            previous = registry.setdefault(key, row.split)
            if previous != row.split:
                raise ValueError(f"split leakage: {key} appears in {previous} and {row.split}")
        if group_families.setdefault(row.group_id, row.family) != row.family:
            raise ValueError("an independent group cannot cross family strata")
        for related in (f"template:{row.template_id}" if row.template_id else "", "prompt:" + digest(" ".join(row.prompt.split()))):
            if related and related_groups.setdefault(related, row.group_id) != row.group_id:
                raise ValueError("related templates or repeated prompts must share one independent group")
        counts[row.split][row.family] += 1
    if require_all_families:
        for split, families in counts.items():
            if set(families) != set(FAMILIES):
                raise ValueError(f"{split} does not cover every product family")
    return {"version": VERSION, "episodes": len(rows), "independent_groups": len(groups), "counts": dict(counts), "corpus_hash": digest([asdict(r) for r in rows])}


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    group_id: str
    family: str
    policy_id: str
    fixture_hash: str
    success: bool | None
    status: str
    cost_usd: float | None
    latency_ms: float
    attempts: int
    model: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    policy_hash: str = ""

    def __post_init__(self) -> None:
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be an explicit boolean or None")
        if any(not isinstance(value, str) or not value for value in (self.episode_id, self.group_id, self.policy_id, self.fixture_hash, self.status)) or self.family not in FAMILIES:
            raise ValueError("outcome identity and status are required")
        if self.cost_usd is not None and (type(self.cost_usd) not in (int, float) or not math.isfinite(self.cost_usd) or self.cost_usd < 0):
            raise ValueError("cost must be known nonnegative finite dollars or None")
        if type(self.latency_ms) not in (int, float) or not math.isfinite(self.latency_ms) or self.latency_ms < 0 or type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("invalid latency/attempt count")
        if self.status != "completed" and self.success is True:
            raise ValueError("a failed or truncated episode cannot count as successful")
        if self.success is True and self.attempts == 0:
            raise ValueError("a successful model episode requires a recorded attempt")


class ProductStudyBudget:
    """Use the existing crash-safe reservation ledger for each fixed stage.

    Stage caps sum to $100. Every pair is reserved in one entry. Interrupted
    trajectories stop collection. Finished unpriced trajectories can retain
    their entire bound while independent episodes use the remaining budget.
    """
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def ledger(self, stage: str) -> StudyBudgetLedger:
        if stage not in STAGE_CAPS:
            raise ValueError("unknown study stage")
        return StudyBudgetLedger(self.directory / f"{stage}.json", StudyBudgetConfig(STAGE_CAPS[stage], 0.0))

    def reserve(self, stage: str, key: str, maximum: float, *, calls: int) -> str:
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("a finite known reservation is required")
        ledger = self.ledger(stage)
        # An interrupted charge blocks the entire study. A finished trajectory
        # may retain its full bound with explicit terminal outcome evidence.
        for name in STAGE_CAPS:
            other = self.ledger(name)
            other.status()  # Validate and create its durable ledger first.
            entries = json.loads(other.path.read_text())["entries"]
            for entry in entries:
                same = name == stage and entry.get("idempotency_key") == key
                if entry.get("actual_cost_usd") is None and not entry.get("terminal_unknown_evidence_hash") and not same:
                    raise StudyBudgetExceeded("unresolved reservation: reconcile before further execution")
                if same and entry.get("actual_cost_usd") is not None:
                    raise StudyBudgetExceeded("settled reservation has no reusable completed pair")
        return ledger.reserve(phase="test" if stage == "test" else "train", model="paired-product-episode", maximum_cost_usd=maximum, idempotency_key=key, potential_api_calls=calls, expected_turns=1, max_turns=calls)

    def retain_terminal_unknown(self, stage: str, reservation: str, evidence_hash: str, *, known_incurred_cost_usd: float) -> None:
        """Keep the entire price bound committed after all trajectories finish.

        This is not settlement. Actual cost stays null, so quality/cost claims
        requiring that observation remain ineligible. Interrupted trajectories
        must not use this path; their reservation continues to stop execution.
        """
        if not isinstance(evidence_hash, str) or len(evidence_hash) != 64:
            raise ValueError("terminal outcome evidence hash required")
        if type(known_incurred_cost_usd) not in (int, float) or not math.isfinite(known_incurred_cost_usd) or known_incurred_cost_usd < 0:
            raise ValueError("finite known charge subtotal required")
        ledger = self.ledger(stage)
        with ledger._locked():
            raw = ledger._load()
            entry = next((e for e in raw["entries"] if e["reservation_id"] == reservation), None)
            if entry is None or entry.get("actual_cost_usd") is not None:
                raise ValueError("an unsettled reservation is required")
            if entry.get("terminal_unknown_evidence_hash") not in (None, evidence_hash):
                raise ValueError("terminal unknown evidence changed")
            if known_incurred_cost_usd > entry["maximum_cost_usd"] + 1e-12:
                raise StudyBudgetExceeded("known charges exceed the episode reservation")
            entry["known_incurred_cost_usd"] = known_incurred_cost_usd
            entry["terminal_unknown_evidence_hash"] = evidence_hash
            ledger._write(raw)

    def settle(self, stage: str, reservation: str, cost: float | None) -> None:
        if cost is None:
            raise StudyBudgetExceeded("unknown charge: reservation retained; reconcile before further execution")
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("settlement requires a known nonnegative finite charge")
        ledger = self.ledger(stage)
        previous = ledger.reservation(reservation)
        if previous is not None and previous.get("actual_cost_usd") is not None:
            if not math.isclose(previous["actual_cost_usd"], cost, abs_tol=1e-12, rel_tol=0):
                raise ValueError("completed charge cannot be changed by reconciliation")
            return
        ledger.settle(reservation, cost)

    def summary(self) -> dict[str, Any]:
        stages = {name: {**self.ledger(name).status(), **self.ledger(name).accounting()} for name in STAGE_CAPS}
        return {"hard_cap_usd": 100.0, "stage_caps_usd": dict(STAGE_CAPS), "stages": stages, "committed_usd": sum(x["committed_usd"] for x in stages.values()), "realized_cost_usd": sum(x["realized_provider_cost_usd"] for x in stages.values()), "outstanding_reserved_cost_usd": sum(x["outstanding_reserved_cost_usd"] for x in stages.values())}


def freeze_test(path: Path, *, episodes: Sequence[Episode], baseline: Mapping[str, Any], challenger: Mapping[str, Any], pricing: Mapping[str, Any], power: Mapping[str, Any], seed: int = 731, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validate_episodes(episodes)
    if not episodes or any(r.split != "test" for r in episodes):
        raise ValueError("a test lock requires exclusively untouched test fixtures")
    rng = random.Random(seed)
    groups: dict[str, list[Episode]] = defaultdict(list)
    for row in episodes:
        groups[row.group_id].append(row)
    ordered = list(groups)
    rng.shuffle(ordered)
    schedule = []
    for group in ordered:
        for row in sorted(groups[group], key=lambda r: r.episode_id):
            order = ["baseline", "challenger"]
            rng.shuffle(order)
            schedule.append({"episode_id": row.episode_id, "fixture_hash": row.content_hash, "policy_order": order})
    payload = {"version": VERSION, "baseline": dict(baseline), "challenger": dict(challenger), "policy_hashes": {"baseline": digest(baseline), "challenger": digest(challenger)}, "pricing": dict(pricing), "power": dict(power), "schedule": schedule, "seed": seed, "family_weights": {f: .2 for f in FAMILIES}, "acceptance": {"quality_lower_one_sided_95_gt": 0, "cost_upper_one_sided_95_lte": 0, "minimum_independent_groups_per_family": MIN_INFERENCE_GROUPS_PER_FAMILY, "nondegenerate_cost_bootstrap_required": True}}
    if protocol is not None:
        payload["protocol"] = dict(protocol)
    payload["lock_hash"] = digest(payload)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise ValueError("test is already frozen with a different policy, sample, or contract")
        return previous
    # O_EXCL prevents accidentally replacing an already published test lock.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def verify_lock(lock: Mapping[str, Any], rows: Sequence[Episode], *, policies: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    unsigned = {k: v for k, v in lock.items() if k != "lock_hash"}
    if digest(unsigned) != lock.get("lock_hash"):
        raise ValueError("test lock content changed")
    validate_episodes(rows)
    if lock.get("version") != VERSION or lock.get("family_weights") != {f: .2 for f in FAMILIES}:
        raise ValueError("test lock has an unsupported measurement contract")
    hashes = {p: digest(lock[p]) for p in ("baseline", "challenger")}
    if hashes != lock.get("policy_hashes") or (policies is not None and {p: digest(policies[p]) for p in hashes} != hashes):
        raise ValueError("policies no longer match the frozen test")
    by_id = {r.episode_id: r for r in rows}
    scheduled = [item["episode_id"] for item in lock["schedule"]]
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != set(by_id):
        raise ValueError("test fixtures no longer match the exact frozen sample")
    for item in lock["schedule"]:
        row = by_id.get(item["episode_id"])
        if row is None or row.content_hash != item["fixture_hash"] or row.split != "test":
            raise ValueError("test fixtures no longer match the frozen sample")
        if sorted(item["policy_order"]) != ["baseline", "challenger"]:
            raise ValueError("invalid frozen policy order")


@contextmanager
def _collector_lock(budget: ProductStudyBudget):
    """A second collector must not spend against an in-flight reservation."""
    import fcntl
    budget.directory.mkdir(parents=True, exist_ok=True)
    with (budget.directory / ".collector.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyBudgetExceeded("another study collector is already executing") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


async def execute_pair(*, row: Episode, policies: Mapping[str, Mapping[str, Any]], order: Sequence[str], runner: Callable[[Episode, str, Mapping[str, Any]], Awaitable[EpisodeOutcome]], maximum_costs: Mapping[str, float], budget: ProductStudyBudget, stage: str, output: Path, lock_hash: str, lock: Mapping[str, Any] | None = None, allow_bounded_unknown: bool = False) -> list[EpisodeOutcome]:
    with _collector_lock(budget):
        return await _execute_pair(row=row, policies=policies, order=order, runner=runner, maximum_costs=maximum_costs, budget=budget, stage=stage, output=output, lock_hash=lock_hash, lock=lock, allow_bounded_unknown=allow_bounded_unknown)


async def _execute_pair(*, row: Episode, policies: Mapping[str, Mapping[str, Any]], order: Sequence[str], runner: Callable[[Episode, str, Mapping[str, Any]], Awaitable[EpisodeOutcome]], maximum_costs: Mapping[str, float], budget: ProductStudyBudget, stage: str, output: Path, lock_hash: str, lock: Mapping[str, Any] | None, allow_bounded_unknown: bool = False) -> list[EpisodeOutcome]:
    if set(order) != {"baseline", "challenger"} or len(order) != 2:
        raise ValueError("a pair must execute each frozen policy exactly once")
    validate_episodes([row], require_all_families=False)
    if set(policies) != set(order) or set(maximum_costs) != set(order) or not lock_hash:
        raise ValueError("both policy configurations and maximum costs are required")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in maximum_costs.values()):
        raise ValueError("each policy requires a finite nonnegative cost bound")
    hashes = {p: digest(policies[p]) for p in order}
    if lock is not None:
        unsigned = {k: v for k, v in lock.items() if k != "lock_hash"}
        scheduled = [item for item in lock["schedule"] if item["episode_id"] == row.episode_id]
        if stage != "test" or digest(unsigned) != lock_hash or lock.get("lock_hash") != lock_hash or lock.get("policy_hashes") != hashes or hashes != {p: digest(lock[p]) for p in hashes}:
            raise ValueError("execution does not match the frozen policy contract")
        if len(scheduled) != 1 or scheduled[0]["fixture_hash"] != row.content_hash or list(order) != scheduled[0]["policy_order"] or row.split != "test":
            raise ValueError("execution does not match the frozen sample and order")
    key = digest([stage, lock_hash, row.episode_id, row.content_hash])
    events = [e for e in read_events(output) if e.get("pair_id") == key]
    started = [e for e in events if e.get("event") == "pair_started"]
    if events and (len(started) != 1 or started[0].get("policy_hashes") != hashes or started[0].get("policy_order") != list(order) or started[0].get("maximum_costs") != dict(maximum_costs)):
        raise ValueError("pair resume changed its policy or execution contract")
    completed = [e for e in events if e.get("event") == "pair_completed"]
    if completed:
        if len(completed) != 1:
            raise ValueError("duplicate terminal pair records")
        results = _terminal_outcomes(started[0], completed[0], events)
        if completed[0]["cost_usd"] is None and allow_bounded_unknown:
            budget.retain_terminal_unknown(stage, completed[0]["reservation_id"], digest([asdict(r) for r in results]), known_incurred_cost_usd=sum(r.cost_usd if r.cost_usd is not None else r.evidence.get("accounting", {}).get("known_total_cost_usd", 0) for r in results))
        else:
            budget.settle(stage, completed[0]["reservation_id"], completed[0]["cost_usd"])
        return results
    if events:
        raise StudyBudgetExceeded("interrupted pair requires reconciliation; refusing to repeat possibly charged work")
    maximum = sum(maximum_costs[p] for p in order)
    reservation = budget.reserve(stage, key, maximum, calls=row.max_calls * 2)
    append_event(output, {"event": "pair_started", "pair_id": key, "reservation_id": reservation, "stage": stage, "episode_id": row.episode_id, "group_id": row.group_id, "family": row.family, "fixture_hash": row.content_hash, "lock_hash": lock_hash, "policy_hashes": hashes, "policy_order": list(order), "maximum_costs": dict(maximum_costs)})
    results = []
    for policy_id in order:
        # The adapter must persist provider attempts as they happen. A timeout
        # at this boundary is intentionally left unresolved, never auto-retried.
        result = await asyncio.wait_for(runner(row, policy_id, policies[policy_id]), timeout=row.max_seconds)
        if digest(policies[policy_id]) != hashes[policy_id] or (result.policy_hash and result.policy_hash != hashes[policy_id]):
            raise ValueError("adapter changed the frozen policy")
        if (result.episode_id, result.group_id, result.family, result.policy_id, result.fixture_hash) != (row.episode_id, row.group_id, row.family, policy_id, row.content_hash):
            raise ValueError("adapter returned an outcome for a different fixture or policy")
        result = replace(result, policy_hash=hashes[policy_id])
        results.append(result)
        append_event(output, {"event": "episode_completed", "pair_id": key, "outcome": asdict(result)})
        if result.cost_usd is None and (not allow_bounded_unknown or result.evidence.get("accounting", {}).get("pending_ids")):
            raise StudyBudgetExceeded("unknown charge: reservation retained; reconcile before further execution")
    known = all(r.cost_usd is not None for r in results)
    total = sum(r.cost_usd or 0 for r in results) if known else None
    # Persist terminal observations before settlement so a crash cannot lose a
    # completed expensive result. Reconciliation can settle the saved pair.
    append_event(output, {"event": "pair_completed", "pair_id": key, "reservation_id": reservation, "stage": stage, "cost_usd": total, "outcomes": [asdict(r) for r in results]})
    if total is None and allow_bounded_unknown:
        budget.retain_terminal_unknown(stage, reservation, digest([asdict(r) for r in results]), known_incurred_cost_usd=sum(r.cost_usd if r.cost_usd is not None else r.evidence.get("accounting", {}).get("known_total_cost_usd", 0) for r in results))
    else:
        budget.settle(stage, reservation, total)
    return results


def _terminal_outcomes(start: Mapping[str, Any], terminal: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> list[EpisodeOutcome]:
    if any(terminal.get(key) != start.get(key) for key in ("pair_id", "reservation_id", "stage")):
        raise ValueError("terminal pair does not match its reservation")
    results = [EpisodeOutcome(**result) for result in terminal["outcomes"]]
    observed = [e["outcome"] for e in events if e.get("event") == "episode_completed"]
    if len(results) != 2 or [r.policy_id for r in results] != start["policy_order"] or observed != terminal["outcomes"]:
        raise ValueError("terminal pair does not match durable episode outcomes")
    for result in results:
        if any(getattr(result, key) != start[key] for key in ("episode_id", "group_id", "family", "fixture_hash")) or result.policy_hash != start["policy_hashes"][result.policy_id]:
            raise ValueError("terminal outcome does not match its fixture or policy")
    total = None if any(r.cost_usd is None for r in results) else sum(r.cost_usd for r in results)
    if terminal.get("cost_usd") != total:
        raise ValueError("terminal pair charge does not equal its observed episode charges")
    return results


def reconcile_completed_pairs(output: Path, budget: ProductStudyBudget) -> dict[str, int]:
    with _collector_lock(budget):
        events = read_events(output)
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            pairs[event["pair_id"]].append(event)
        complete = 0
        for records in pairs.values():
            starts = [e for e in records if e.get("event") == "pair_started"]
            terminals = [e for e in records if e.get("event") == "pair_completed"]
            if len(starts) != 1 or len(terminals) > 1:
                raise ValueError("missing or duplicate pair lifecycle records")
            start = starts[0]
            if not terminals:
                observed = [e["outcome"] for e in records if e.get("event") == "episode_completed"]
                if len(observed) != 2:
                    continue
                results = [EpisodeOutcome(**r) for r in observed]
                terminal = {"event": "pair_completed", "pair_id": start["pair_id"], "reservation_id": start["reservation_id"], "stage": start["stage"], "cost_usd": None if any(r.cost_usd is None for r in results) else sum(r.cost_usd for r in results), "outcomes": observed}
                _terminal_outcomes(start, terminal, records)
                append_event(output, terminal)
            else:
                terminal = terminals[0]
                _terminal_outcomes(start, terminal, records)
            if terminal["cost_usd"] is not None:
                budget.settle(terminal["stage"], terminal["reservation_id"], terminal["cost_usd"])
                complete += 1
        return {"completed": complete, "unresolved": len(pairs) - complete}


def _quantile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo) if lo != hi else ordered[lo]


def paired_report(rows: Sequence[Episode], outcomes: Sequence[EpisodeOutcome], *, seed: int = 731, bootstrap_samples: int = 4000, lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validate_episodes(rows)
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if lock is not None:
        verify_lock(lock, rows)
    expected = {r.episode_id: r for r in rows}
    indexed: dict[tuple[str, str], EpisodeOutcome] = {}
    problems: list[str] = []
    for result in outcomes:
        key = result.episode_id, result.policy_id
        row = expected.get(result.episode_id)
        if key in indexed:
            problems.append(f"duplicate:{key}")
        if row is None or result.policy_id not in {"baseline", "challenger"}:
            problems.append(f"unexpected:{key}")
        elif (result.fixture_hash, result.family, result.group_id) != (row.content_hash, row.family, row.group_id):
            problems.append(f"identity_mismatch:{key}")
        elif lock is not None and result.policy_hash != lock["policy_hashes"][result.policy_id]:
            problems.append(f"unfrozen_policy:{key}")
        indexed[key] = result
    families: dict[str, list[tuple[EpisodeOutcome, EpisodeOutcome]]] = defaultdict(list)
    for row in rows:
        baseline = indexed.get((row.episode_id, "baseline"))
        challenger = indexed.get((row.episode_id, "challenger"))
        if baseline is None or challenger is None:
            problems.append(f"missing_pair:{row.episode_id}")
            continue
        if any(r.success is None or r.cost_usd is None for r in (baseline, challenger)):
            problems.append(f"ungradable_or_unknown_cost:{row.episode_id}")
            continue
        families[row.family].append((baseline, challenger))
    report: dict[str, Any] = {"version": VERSION, "comparison": "actual_current_route_vs_one_frozen_challenger", "population": "balanced_public_and_authored_product_fixtures", "status": "incomplete" if problems else "inconclusive", "problems": problems, "inference_limitations": [] if lock is not None else ["no_verified_test_lock"], "lock_hash": lock["lock_hash"] if lock is not None else None, "expected_episodes": len(rows), "observed_outcomes": len(outcomes), "known_recorded_cost_usd": sum(r.cost_usd for r in outcomes if r.cost_usd is not None), "unknown_cost_outcomes": sum(r.cost_usd is None for r in outcomes), "operational_failures": sum(r.status != "completed" for r in outcomes), "attempts": sum(r.attempts for r in outcomes), "family_weights": {f: .2 for f in FAMILIES}, "families": {}}
    cluster_pairs = {}
    for family in FAMILIES:
        pairs = families[family]
        if not pairs:
            report["problems"].append(f"empty_family:{family}")
            continue
        clusters: dict[str, list[tuple[EpisodeOutcome, EpisodeOutcome]]] = defaultdict(list)
        for pair in pairs:
            clusters[pair[0].group_id].append(pair)
        cluster_pairs[family] = list(clusters.values())
        summary = {"episodes": len(pairs), "independent_groups": len(clusters)}
        for index, label in enumerate(("baseline", "challenger")):
            records = [p[index] for p in pairs]
            summary[label] = {"success": statistics.mean(float(r.success) for r in records), "mean_cost_usd": statistics.mean(r.cost_usd for r in records), "latency_p50_ms": _quantile([r.latency_ms for r in records], .5), "latency_p95_ms": _quantile([r.latency_ms for r in records], .95), "operational_failures": sum(r.status != "completed" for r in records), "attempts": sum(r.attempts for r in records)}
        report["families"][family] = summary
        if len(clusters) < 2:
            report["problems"].append(f"insufficient_independent_groups:{family}")
        if len(clusters) < MIN_INFERENCE_GROUPS_PER_FAMILY:
            report["inference_limitations"].append(f"too_few_clusters_for_bootstrap_inference:{family}:{len(clusters)}<{MIN_INFERENCE_GROUPS_PER_FAMILY}")
    if report["problems"]:
        report["status"] = "incomplete"
        return report
    def differences(groups):
        quality, costs = [], []
        for family in FAMILIES:
            pairs = [pair for group in groups[family] for pair in group]
            quality.append(statistics.mean(float(b.success) - float(a.success) for a, b in pairs))
            costs.append(statistics.mean(b.cost_usd - a.cost_usd for a, b in pairs))
        return statistics.mean(quality), statistics.mean(costs)
    delta_q, delta_c = differences(cluster_pairs)
    rng = random.Random(seed)
    samples = [differences({f: rng.choices(cluster_pairs[f], k=len(cluster_pairs[f])) for f in FAMILIES}) for _ in range(bootstrap_samples)]
    lower_q = _quantile([x[0] for x in samples], .05)
    upper_c = _quantile([x[1] for x in samples], .95)
    if math.isclose(min(x[1] for x in samples), max(x[1] for x in samples), abs_tol=1e-15, rel_tol=0):
        report["inference_limitations"].append("degenerate_cost_bootstrap_cannot_estimate_cost_uncertainty")
    quality_gate, cost_gate = lower_q > 0, upper_c <= 0
    report.update({"status": "improved" if quality_gate and cost_gate and not report["inference_limitations"] else "inconclusive", "success_difference": delta_q, "mean_cost_difference_usd": delta_c, "success_lower_one_sided_95": lower_q, "cost_upper_one_sided_95_usd": upper_c, "quality_gate": quality_gate, "cost_gate": cost_gate, "bootstrap": {"method": "paired_stratified_group_percentile", "samples": bootstrap_samples, "seed": seed, "minimum_independent_groups_per_family": MIN_INFERENCE_GROUPS_PER_FAMILY, "bounds_are_diagnostic_only": bool(report["inference_limitations"])}, "interpretation": "Aggregate evidence does not establish improvement in every family or represent production traffic weights. Small-sample or degenerate bootstrap bounds are diagnostics, not evidence of improvement."})
    return report


def estimate_power(*, pilot: Sequence[EpisodeOutcome], available_test_usd: float = 50.0, target_lift: float = .05, conservative_pair_cost: float | None = None) -> dict[str, Any]:
    """Plan independent groups, retaining uncertainty from a small pilot.

    Paired quality differences lie in [-1,1]. Their second moments bound
    variance; a one-sided Hoeffding cushion prevents a tiny perfect pilot from
    claiming zero variability. Repeated trials share a cluster and never add
    independent groups. This remains a planning approximation, not achieved
    power, especially when the final challenger differs from the pilot policy.
    """
    if not math.isfinite(target_lift) or not 0 < target_lift <= 1 or not math.isfinite(available_test_usd) or available_test_usd < 0:
        raise ValueError("invalid effect or available budget")
    if conservative_pair_cost is not None and (not math.isfinite(conservative_pair_cost) or conservative_pair_cost <= 0):
        raise ValueError("explicit conservative pair cost must be positive")
    by_id: dict[str, dict[str, EpisodeOutcome]] = defaultdict(dict)
    for result in pilot:
        if result.policy_id not in {"baseline", "challenger"} or result.policy_id in by_id[result.episode_id]:
            raise ValueError("invalid or duplicate pilot policy outcome")
        by_id[result.episode_id][result.policy_id] = result
    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    costs, incomplete = [], 0
    for pair in by_id.values():
        if set(pair) != {"baseline", "challenger"}:
            incomplete += 1
            continue
        a, b = pair["baseline"], pair["challenger"]
        if (a.group_id, a.family, a.fixture_hash) != (b.group_id, b.family, b.fixture_hash):
            raise ValueError("pilot pair identities disagree")
        if a.success is None or b.success is None or a.cost_usd is None or b.cost_usd is None:
            incomplete += 1
            continue
        differences[(a.family, a.group_id)].append(float(b.success) - float(a.success))
        costs.append(a.cost_usd + b.cost_usd)
    cluster_values = [statistics.mean(values) for values in differences.values()]
    n = len(cluster_values)
    second_moment = statistics.mean(value * value for value in cluster_values) if n else 1.0
    variance = max(.25, min(1.0, second_moment + math.sqrt(math.log(20) / (2 * n)))) if n else 1.0
    pairs_per_group = max(1, math.ceil(len(costs) / max(n, 1)))
    z = statistics.NormalDist().inv_cdf(.95) + statistics.NormalDist().inv_cdf(.8)
    needed_groups = math.ceil((z * z * variance / target_lift**2) / 5) * 5
    pair_cost = conservative_pair_cost if conservative_pair_cost is not None else (2 * statistics.mean(costs) if costs else None)
    affordable_pairs = math.floor(available_test_usd / pair_cost / 5) * 5 if pair_cost and pair_cost > 0 else None
    affordable_groups = math.floor(affordable_pairs / pairs_per_group / 5) * 5 if affordable_pairs is not None else None
    missing = sorted(set(FAMILIES) - {family for family, _ in differences})
    complete = not missing and not incomplete and bool(n)
    return {"method": "conservative_paired_group_normal_approximation", "target_absolute_lift": target_lift,
        "target_power": .8, "one_sided_alpha": .05, "pilot_pairs": len(costs), "pilot_groups": n,
        "incomplete_pilot_pairs": incomplete, "missing_pilot_families": missing, "pilot_complete": complete,
        "variance_floor": .25, "used_variance": variance, "variance_method": "bounded_second_moment_with_95_percent_Hoeffding_cushion",
        "cluster_inflation": pairs_per_group, "required_independent_groups": needed_groups,
        "required_pairs": needed_groups * pairs_per_group, "conservative_pair_cost_usd": pair_cost,
        "affordable_pairs": affordable_pairs, "affordable_independent_groups": affordable_groups,
        "budget_supports_target": bool(complete and affordable_groups is not None and affordable_groups >= needed_groups),
        "approximate_detectable_effect_at_affordable_n": z * math.sqrt(variance / affordable_groups) if affordable_groups else None,
        "caveat": "Independent task availability can constrain the sample below budget capacity. Pilot and final policies may differ; this is not achieved power or improvement evidence."}
