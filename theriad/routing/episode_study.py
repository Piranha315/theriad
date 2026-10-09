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
VALID_SPLITS = {"pilot", "train", "calibration", "threshold_selection", "validation", "test", "smoke"}
MIN_INFERENCE_GROUPS_PER_FAMILY = 20
# The product runtime enforces row.max_seconds for task/provider work. The
# collector watchdog must leave bounded time for cancellation accounting and
# the terminal outcome; an equal deadline cancels that finalization first.
OUTCOME_FINALIZATION_GRACE_SECONDS = 5.0


async def run_episode_with_finalization(runner, row, policy_id, policy):
    return await asyncio.wait_for(
        runner(row, policy_id, policy),
        timeout=row.max_seconds + OUTCOME_FINALIZATION_GRACE_SECONDS,
    )


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
    from .episode_fixtures import known_template_lineage

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
        if row.family not in FAMILIES or row.split not in VALID_SPLITS:
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
        for related in (f"template:{row.template_id}" if row.template_id else "", "prompt:" + digest(" ".join(row.prompt.split())), *known_template_lineage(row)):
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
    def __init__(self, directory: str | Path, config: Mapping[str, Any] | None = None):
        self.directory = Path(directory)
        path = self.directory.parent / "budget-config.json"
        if config is None and path.exists():
            config = json.loads(path.read_text())
        config = dict(config or {})
        stage_caps = config.get("stage_caps_usd", STAGE_CAPS)
        if not isinstance(stage_caps, Mapping) or set(stage_caps) != set(STAGE_CAPS) or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in stage_caps.values()):
            raise ValueError("budget config must provide finite nonnegative caps for every study stage")
        self.stage_caps = {name: float(stage_caps[name]) for name in STAGE_CAPS}
        hard_cap = config.get("hard_cap_usd", sum(self.stage_caps.values()))
        if type(hard_cap) not in (int, float) or not math.isfinite(hard_cap) or hard_cap < 0 or sum(self.stage_caps.values()) > hard_cap + 1e-12:
            raise ValueError("finite nonnegative hard cap must cover configured stage caps")
        self.hard_cap_usd = float(hard_cap)

    def ledger(self, stage: str) -> StudyBudgetLedger:
        if stage not in self.stage_caps:
            raise ValueError("unknown study stage")
        return StudyBudgetLedger(self.directory / f"{stage}.json", StudyBudgetConfig(self.stage_caps[stage], 0.0))

    def reserve(self, stage: str, key: str, maximum: float, *, calls: int) -> str:
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("a finite known reservation is required")
        ledger = self.ledger(stage)
        # An interrupted charge blocks the entire study. A finished trajectory
        # may retain its full bound with explicit terminal outcome evidence.
        for name in self.stage_caps:
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
        stages = {name: {**self.ledger(name).status(), **self.ledger(name).accounting()} for name in self.stage_caps}
        return {"hard_cap_usd": self.hard_cap_usd, "stage_caps_usd": dict(self.stage_caps), "stages": stages, "committed_usd": sum(x["committed_usd"] for x in stages.values()), "realized_cost_usd": sum(x["realized_provider_cost_usd"] for x in stages.values()), "outstanding_reserved_cost_usd": sum(x["outstanding_reserved_cost_usd"] for x in stages.values())}


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
    payload = {"version": VERSION, "baseline": dict(baseline), "challenger": dict(challenger), "policy_hashes": {"baseline": digest(baseline), "challenger": digest(challenger)}, "pricing": dict(pricing), "power": dict(power), "schedule": schedule, "seed": seed, "family_weights": {f: .2 for f in FAMILIES}, "acceptance": {"quality_lower_one_sided_95_gt": -0.01, "cost_per_success_ratio_upper_one_sided_95_lt": 0.90, "minimum_independent_groups_per_family": MIN_INFERENCE_GROUPS_PER_FAMILY, "nondegenerate_cost_bootstrap_required": True}}
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
        result = await run_episode_with_finalization(runner, row, policy_id, policies[policy_id])
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
    if lo == hi or ordered[lo] == ordered[hi]:
        return ordered[lo]
    # Interpolating infinity with zero weight produces NaN; an unbounded
    # upper observation must remain unbounded, including at the 95th percentile.
    if not math.isfinite(ordered[hi]):
        return ordered[hi]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


# Each record is one independent group's episode means: baseline/challenger
# success and baseline/challenger cost. All families and groups get equal weight.
GroupValues = Mapping[str, Sequence[tuple[float, float, float, float]]]


def _paired_group_metrics(groups: GroupValues) -> tuple[float, float, float, float, float, float]:
    means = [tuple(sum(values) / len(records) for values in zip(*records))
             for records in (groups[family] for family in FAMILIES)]
    baseline_q, challenger_q, baseline_cost, challenger_cost = (
        sum(values) / len(FAMILIES) for values in zip(*means)
    )
    ratio = ((challenger_cost / challenger_q) / (baseline_cost / baseline_q)
             if baseline_q > 0 and challenger_q > 0 and baseline_cost > 0 else math.inf)
    return challenger_q - baseline_q, ratio, baseline_cost, challenger_cost, baseline_q, challenger_q


def _paired_bootstrap_decision(groups: GroupValues, *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    """The same joint statistical decision for real reports and synthetic studies.

    Pairing is retained within a group and families are resampled separately.
    Lock verification and complete-accounting checks belong to paired_report;
    simulations assume those gates have been satisfied by design.
    """
    rng = random.Random(seed)
    point = _paired_group_metrics(groups)
    samples = [_paired_group_metrics({family: rng.choices(groups[family], k=len(groups[family]))
                                      for family in FAMILIES}) for _ in range(bootstrap_samples)]
    quality = [sample[0] for sample in samples]
    ratios = [sample[1] for sample in samples]
    lower, upper = _quantile(quality, .05), _quantile(ratios, .95)
    limitations = []
    if not math.isfinite(point[1]) or not math.isfinite(upper):
        limitations.append("degenerate_or_zero_success_cost_ratio")
    if math.isclose(max(quality), min(quality), rel_tol=1e-12, abs_tol=1e-12):
        limitations.append("degenerate_quality_bootstrap")
    if math.isclose(max(ratios), min(ratios), rel_tol=1e-12, abs_tol=1e-12):
        limitations.append("degenerate_cost_ratio_bootstrap")
    enough_groups = all(len(groups[family]) >= MIN_INFERENCE_GROUPS_PER_FAMILY for family in FAMILIES)
    quality_gate = lower > -.01 and not math.isclose(lower, -.01, rel_tol=0, abs_tol=1e-12)
    cost_gate = math.isfinite(upper) and upper < .90 and not math.isclose(upper, .90, rel_tol=0, abs_tol=1e-12)
    return {"point": point, "quality_lower": lower, "cost_ratio_upper": upper,
            "quality_gate": quality_gate, "cost_gate": cost_gate, "limitations": limitations,
            "undefined_cost_ratio_resamples": sum(not math.isfinite(value) for value in ratios),
            "joint_pass": quality_gate and cost_gate and enough_groups and not limitations}


@dataclass(frozen=True)
class JointPowerScenario:
    """Explicit synthetic population assumptions, never estimates from test data.

    Discordance is P(baseline success != challenger success), fixing the entire
    paired Bernoulli law given the two marginal probabilities. Costs are paired
    lognormal draws independent of success; shared_cost_fraction specifies the
    shared fraction of log-cost variance. Means target the population ratio of
    mean cost per success, rather than a mean of per-episode ratios. Repeats in
    one group share all their outcomes (perfect within-group dependence).
    """
    name: str
    baseline_success_probability: float = .70
    challenger_success_probability: float = .70
    success_discordance_probability: float = .20
    cost_per_success_ratio: float = .75
    cost_cv: float = .50
    shared_cost_fraction: float = .50
    episodes_per_group: int = 1
    baseline_mean_cost_usd: float = .02

    def __post_init__(self) -> None:
        numeric = (self.baseline_success_probability, self.challenger_success_probability,
                   self.success_discordance_probability, self.cost_per_success_ratio,
                   self.cost_cv, self.shared_cost_fraction, self.baseline_mean_cost_usd)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric):
            raise ValueError("scenario assumptions must be finite numbers")
        a, b, d = numeric[:3]
        if not self.name or not isinstance(self.name, str) or not (0 < a <= 1 and 0 < b <= 1):
            raise ValueError("scenario needs a name and positive success probabilities")
        if not 0 <= d <= 1 or not abs(a - b) - 1e-12 <= d <= min(a + b, 2 - a - b) + 1e-12:
            raise ValueError("discordance is incompatible with the marginal success probabilities")
        if self.cost_per_success_ratio <= 0 or not 0 <= self.cost_cv <= 5 or not 0 <= self.shared_cost_fraction <= 1 or self.baseline_mean_cost_usd <= 0:
            raise ValueError("invalid positive costs, cost CV (0..5), or shared fraction (0..1)")
        if type(self.episodes_per_group) is not int or self.episodes_per_group < 1:
            raise ValueError("episodes_per_group must be a positive integer")


DEFAULT_JOINT_POWER_SCENARIOS = (
    JointPowerScenario("alternative_equal_quality_25pct_saving"),
    JointPowerScenario("alternative_plus2pp_quality_20pct_saving", challenger_success_probability=.72, cost_per_success_ratio=.80),
    JointPowerScenario("quality_boundary_null", challenger_success_probability=.69),
    JointPowerScenario("cost_boundary_null", cost_per_success_ratio=.90),
    JointPowerScenario("both_boundaries_null", challenger_success_probability=.69, cost_per_success_ratio=.90),
)


def _synthetic_group_values(scenario: JointPowerScenario, groups_per_family: int | Mapping[str, int], rng: random.Random) -> dict[str, list[tuple[float, float, float, float]]]:
    a, b, discordance = scenario.baseline_success_probability, scenario.challenger_success_probability, scenario.success_discordance_probability
    both = (a + b - discordance) / 2
    sigma = math.sqrt(math.log1p(scenario.cost_cv ** 2))
    shared_scale, individual_scale = math.sqrt(scenario.shared_cost_fraction), math.sqrt(1 - scenario.shared_cost_fraction)
    challenger_mean = scenario.baseline_mean_cost_usd * scenario.cost_per_success_ratio * b / a
    groups = {}
    for family in FAMILIES:
        records = []
        count = groups_per_family[family] if isinstance(groups_per_family, Mapping) else groups_per_family
        for _ in range(count):
            draw = rng.random()
            baseline_success = draw < a
            challenger_success = draw < both or a <= draw < a + b - both
            shared = rng.gauss(0, 1)
            costs = [mean * math.exp(sigma * (shared_scale * shared + individual_scale * rng.gauss(0, 1)) - .5 * sigma ** 2)
                     for mean in (scenario.baseline_mean_cost_usd, challenger_mean)]
            records.append((float(baseline_success), float(challenger_success), *costs))
        groups[family] = records
    return groups


def _monte_carlo_rate(count: int, repetitions: int) -> dict[str, Any]:
    """Wilson 95% interval remains informative when no passes are observed."""
    proportion = count / repetitions
    z = statistics.NormalDist().inv_cdf(.975)
    denominator = 1 + z * z / repetitions
    center = (proportion + z * z / (2 * repetitions)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / repetitions + z * z / (4 * repetitions ** 2)) / denominator
    return {"count": count, "estimate": proportion, "monte_carlo_standard_error": math.sqrt(proportion * (1 - proportion) / repetitions),
            "wilson_95_interval": [max(0., center - radius), min(1., center + radius)]}


def simulate_joint_power(*, groups_per_family: int = 20, scenarios: Sequence[JointPowerScenario] | None = None,
                         repetitions: int = 100, bootstrap_samples: int = 400, seed: int = 731,
                         group_counts_by_family: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Bounded scenario-based joint power/false-pass simulation, entirely offline.

    This calls the actual paired, family-stratified group bootstrap decision.
    Monte Carlo intervals quantify simulation error only, not uncertainty that
    the assumed synthetic population describes future product traffic.
    """
    if type(groups_per_family) is not int or groups_per_family < 2:
        raise ValueError("simulation needs at least two groups per family")
    if group_counts_by_family is not None:
        if (set(group_counts_by_family) != set(FAMILIES)
                or any(type(n) is not int or n < 1 for n in group_counts_by_family.values())):
            raise ValueError("explicit group counts must cover every family with positive integers")
        group_counts_by_family = dict(group_counts_by_family)
    if type(repetitions) is not int or repetitions < 1 or type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("positive repetitions and at least 100 bootstrap samples required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    scenarios = tuple(DEFAULT_JOINT_POWER_SCENARIOS if scenarios is None else scenarios)
    if not scenarios or any(not isinstance(item, JointPowerScenario) for item in scenarios) or len({item.name for item in scenarios}) != len(scenarios):
        raise ValueError("provide uniquely named JointPowerScenario objects")
    results = []
    for scenario in scenarios:
        # Stable per-scenario streams allow comparisons with a subset or a reordered list.
        scenario_seed = int(digest({"seed": seed, "scenario": asdict(scenario)})[:16], 16)
        rng = random.Random(scenario_seed)
        counts = {"joint_pass": 0, "quality_gate": 0, "cost_gate": 0, "degenerate_inference": 0}
        for _ in range(repetitions):
            groups = _synthetic_group_values(scenario, group_counts_by_family if group_counts_by_family is not None else groups_per_family, rng)
            result = _paired_bootstrap_decision(groups, seed=rng.randrange(1 << 63), bootstrap_samples=bootstrap_samples)
            for key in ("joint_pass", "quality_gate", "cost_gate"):
                counts[key] += int(result[key])
            counts["degenerate_inference"] += int(bool(result["limitations"]))
        delta = scenario.challenger_success_probability - scenario.baseline_success_probability
        # Decimal input at the exact boundary must remain a null despite float rounding.
        alternative = delta > -.01 and not math.isclose(delta, -.01, rel_tol=0, abs_tol=1e-12) and scenario.cost_per_success_ratio < .90
        results.append({"scenario": asdict(scenario), "scenario_seed": scenario_seed,
                        "population_success_difference": delta, "population_cost_per_success_ratio": scenario.cost_per_success_ratio,
                        "hypothesis": "joint_alternative" if alternative else "joint_null",
                        "joint_rate_interpretation": "scenario_joint_power" if alternative else "scenario_false_pass_rate",
                        "rates": {key: _monte_carlo_rate(value, repetitions) for key, value in counts.items()}})
    return {"method": "scenario_based_paired_family_stratified_group_bootstrap_joint_power",
            "groups_per_family": groups_per_family if group_counts_by_family is None else None,
            "group_counts_by_family": group_counts_by_family,
            "independent_groups": groups_per_family * len(FAMILIES) if group_counts_by_family is None else sum(group_counts_by_family.values()),
            "repetitions": repetitions, "bootstrap_samples": bootstrap_samples, "seed": seed,
            "acceptance": {"quality_lower_one_sided_95_gt": -.01, "cost_per_success_ratio_upper_one_sided_95_lt": .90,
                           "minimum_independent_groups_per_family": MIN_INFERENCE_GROUPS_PER_FAMILY, "nondegenerate_bootstrap_required": True},
            "results": results, "achieved_empirical_power": None,
            "assumptions": ["Equal family weights and equal independent group weights within each family.",
                            "IID groups within and across families; paired Bernoulli successes use the stated discordance.",
                            "Paired lognormal costs are independent of success; cost parameters are identical across families.",
                            "Within-group repeats are perfectly dependent and add no independent sample size.",
                            "Complete known costs, gradable outcomes, and valid policy lock are assumed; collection failures reduce real acceptance.",
                            "Wilson intervals cover Monte Carlo error only; neither scenario realism nor bootstrap coverage is guaranteed."],
            "interpretation": "Synthetic scenario-based joint power and null false-pass diagnostics only. No achieved empirical power, pilot-calibrated forecast, or promotion evidence is claimed."}


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
    group_values = {
        family: [tuple(statistics.mean(float(getattr(pair[index], field)) for pair in group)
                       for index, field in ((0, "success"), (1, "success"), (0, "cost_usd"), (1, "cost_usd")))
                 for group in cluster_pairs[family]]
        for family in FAMILIES
    }
    decision = _paired_bootstrap_decision(group_values, seed=seed, bootstrap_samples=bootstrap_samples)
    delta_q, cost_ratio, baseline_cost, challenger_cost, baseline_success, challenger_success = decision["point"]
    lower_q, upper_ratio = decision["quality_lower"], decision["cost_ratio_upper"]
    report["inference_limitations"].extend(decision["limitations"])
    quality_gate, cost_gate = decision["quality_gate"], decision["cost_gate"]
    report.update({
        "status": "improved" if quality_gate and cost_gate and not report["inference_limitations"] else "inconclusive",
        "success_difference": delta_q,
        "mean_cost_difference_usd": challenger_cost - baseline_cost,
        "success_lower_one_sided_95": lower_q,
        "cost_per_success_ratio": cost_ratio if math.isfinite(cost_ratio) else None,
        "cost_upper_one_sided_95_usd": None,
        "cost_per_success_ratio_upper_one_sided_95": upper_ratio if math.isfinite(upper_ratio) else None,
        "baseline_cost_per_success_usd": baseline_cost / baseline_success if baseline_success > 0 else None,
        "challenger_cost_per_success_usd": challenger_cost / challenger_success if challenger_success > 0 else None,
        "quality_gate": quality_gate,
        "cost_gate": cost_gate,
        "acceptance": {"quality_lower_bound_gt": -0.01, "cost_ratio_upper_bound_lt": 0.90},
        "bootstrap": {"method": "paired_family_stratified_group_percentile", "samples": bootstrap_samples, "seed": seed, "minimum_independent_groups_per_family": MIN_INFERENCE_GROUPS_PER_FAMILY, "undefined_cost_ratio_resamples": decision["undefined_cost_ratio_resamples"], "bounds_are_diagnostic_only": bool(report["inference_limitations"])},
        "interpretation": "Offline evidence is benchmark-weighted, not production-weighted. A pass requires complete accounting, valid inference, quality noninferiority, and cost per successful episode below the locked ratio gate.",
    })
    return report


def simulate_inference_operating_characteristics(
    *, groups_per_family: int = 20, repetitions: int = 100, seed: int = 731
) -> dict[str, Any]:
    """Stress the locked gates without provider calls or empirical claims.

    The null simulation uses equal success probabilities and equal costs. It
    reports bootstrap false-pass frequency and lower-bound coverage only as an
    implementation diagnostic; it is not a power estimate for a future study.
    """
    if groups_per_family < MIN_INFERENCE_GROUPS_PER_FAMILY or repetitions < 1:
        raise ValueError("simulation needs at least the minimum independent groups")
    rng = random.Random(seed)
    false_passes = 0
    covered_quality = 0
    covered_ratio = 0
    for _ in range(repetitions):
        rows: list[Episode] = []
        outcomes: list[EpisodeOutcome] = []
        for family in FAMILIES:
            for group in range(groups_per_family):
                row = Episode(f"{family}-{group}", f"{family}-group-{group}", family, "test", f"simulation {family} {group}", {}, {"kind": "exact", "answer": "yes"})
                rows.append(row)
                baseline_success = bool(rng.getrandbits(1))
                challenger_success = bool(rng.getrandbits(1))
                outcomes.extend((
                    EpisodeOutcome(row.episode_id, row.group_id, family, "baseline", row.content_hash, baseline_success, "completed", 0.02, 100, 1),
                    EpisodeOutcome(row.episode_id, row.group_id, family, "challenger", row.content_hash, challenger_success, "completed", 0.02, 100, 1),
                ))
        diagnostic = paired_report(rows, outcomes, seed=rng.randrange(1 << 30), bootstrap_samples=100)
        lower = diagnostic.get("success_lower_one_sided_95")
        ratio = diagnostic.get("cost_per_success_ratio_upper_one_sided_95")
        covered_quality += bool(lower is not None and lower <= 0.0)
        # A null serialized upper ratio denotes an unbounded interval, which
        # includes the population ratio; it is never evidence for a pass.
        covered_ratio += bool(ratio is None or ratio >= 1.0)
        false_passes += int(bool(diagnostic.get("quality_gate") and diagnostic.get("cost_gate")))
    return {
        "method": "paired_family_stratified_group_bootstrap_null_simulation",
        "groups_per_family": groups_per_family,
        "repetitions": repetitions,
        "quality_lower_bound_coverage": covered_quality / repetitions,
        "cost_ratio_upper_bound_coverage": covered_ratio / repetitions,
        "false_pass_rate_under_equal_quality_and_cost": false_passes / repetitions,
        "interpretation": "diagnostic operating characteristics only; no provider calls and no future-study power claim",
    }


def estimate_power(*, pilot: Sequence[EpisodeOutcome], available_test_usd: float = 44.0, target_lift: float | None = None, conservative_pair_cost: float | None = None) -> dict[str, Any]:
    """Plan independent groups, retaining uncertainty from a small pilot.

    Paired quality differences lie in [-1,1]. Their second moments bound
    variance; a one-sided Hoeffding cushion prevents a tiny perfect pilot from
    claiming zero variability. Repeated trials share a cluster and never add
    independent groups. This remains a planning approximation, not achieved
    power, especially when the final challenger differs from the pilot policy.
    """
    if not math.isfinite(available_test_usd) or available_test_usd < 0:
        raise ValueError("invalid effect or available budget")
    quality_margin = 0.01 if target_lift is None else target_lift
    if not math.isfinite(quality_margin) or not 0 < quality_margin <= 1:
        raise ValueError("invalid quality boundary margin")
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
    needed_groups = math.ceil((z * z * variance / quality_margin**2) / 5) * 5
    pair_cost = conservative_pair_cost if conservative_pair_cost is not None else (2 * statistics.mean(costs) if costs else None)
    affordable_pairs = math.floor(available_test_usd / pair_cost / 5) * 5 if pair_cost and pair_cost > 0 else None
    affordable_groups = math.floor(affordable_pairs / pairs_per_group / 5) * 5 if affordable_pairs is not None else None
    missing = sorted(set(FAMILIES) - {family for family, _ in differences})
    complete = not missing and not incomplete and bool(n)
    return {"method": "quality_only_normal_boundary_distance_planning_approximation", "target_absolute_lift": None,
        "scope": "quality_only_sample_size_approximation", "joint_power_estimate": None, "achieved_empirical_power": None,
        "joint_power_target": None, "quality_only_nominal_power_target": .8, "one_sided_alpha": .05, "quality_margin": -0.01,
        "cost_per_success_ratio_upper_bound": .90, "boundary_margin_used_for_approximation": quality_margin,
        "pilot_pairs": len(costs), "pilot_groups": n,
        "incomplete_pilot_pairs": incomplete, "missing_pilot_families": missing, "pilot_complete": complete,
        "variance_floor": .25, "used_variance": variance, "variance_method": "bounded_second_moment_with_95_percent_Hoeffding_cushion",
        "cluster_inflation": pairs_per_group, "required_independent_groups": needed_groups,
        "required_pairs": needed_groups * pairs_per_group, "conservative_pair_cost_usd": pair_cost,
        "affordable_pairs": affordable_pairs, "affordable_independent_groups": affordable_groups,
        "budget_supports_target": False,
        "budget_supports_quality_only_approximation": bool(complete and affordable_groups is not None and affordable_groups >= needed_groups),
        "approximate_detectable_effect_at_affordable_n": z * math.sqrt(variance / affordable_groups) if affordable_groups else None,
        "caveat": "Required groups approximate only the quality gate using a nominal normal calculation and a conservative variance heuristic, not the joint bootstrap decision. budget_supports_target stays false because joint power is unassessed. Use simulate_joint_power with explicit quality, discordance, and cost assumptions for synthetic joint power and boundary-null false-pass diagnostics. No achieved empirical power is claimed."}
