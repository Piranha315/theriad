"""Luna compatibility amendment planning on the original episode study.

This module selects original development identities; it does not replace the
canonical collector, rewrite historical attempts, or authorize held-out reads.
"""
from dataclasses import asdict
import math
import hashlib
from pathlib import Path

from theriad.core.providers.openrouter_capabilities import LUNA
from .episode_study import digest


def missing_luna_cells(rows, outcomes, events):
    """Separate absent outcomes from the two unresolved dispatch failures.

    Observed task failures are retained, not selected for quality-driven retry.
    Unknown charges cannot be treated as absent observations or cleared here.
    """
    if any(row.split not in {"train", "calibration", "threshold_selection"} for row in rows):
        raise ValueError("Luna amendment is limited to original development splits")
    if len({r.episode_id for r in rows}) != len(rows):
        raise ValueError("duplicate original episode identity")
    observed = {}
    for value in outcomes:
        if value.policy_id == LUNA:
            if value.episode_id in observed:
                raise ValueError("duplicate Luna outcome")
            observed[value.episode_id] = value
    terminals = [e for e in events if e.get("event") == "attempt_finished" and e.get("model") == LUNA]
    starts = {e["attempt_id"] for e in events if e.get("event") == "attempt_started" and e.get("model") == LUNA}
    if starts - {e["attempt_id"] for e in terminals}:
        raise ValueError("pending Luna attempts must be resolved before planning")
    missing, blocked, retained = [], [], []
    for row in rows:
        identity = {"episode_id": row.episode_id, "group_id": row.group_id,
                    "family": row.family, "split": row.split, "original_row_hash": row.content_hash}
        previous = observed.get(row.episode_id)
        attempts = [e for e in terminals if e.get("episode_id") == row.episode_id]
        if previous is None:
            if attempts:
                raise ValueError("attempted Luna cell without an outcome needs recovery, not a fresh call")
            missing.append(identity)
        elif previous.status == "execution_error" and attempts and all(e.get("provider_status_code") == 404 and e.get("cost_usd") is None for e in attempts):
            blocked.append({**identity, "original_outcome_hash": digest(asdict(previous)),
                            "attempts": [{"attempt_id": e["attempt_id"], "terminal_hash": digest(e),
                                          "recorded_at": e["recorded_at"]} for e in attempts],
                            "reason": "provider billing reconciliation required before retry"})
        else:
            retained.append(identity)
    return {"missing": missing, "billing_blocked": blocked, "retained": retained,
            "collection_allowed": False,
            "note": "Inventory only. Entire Luna continuation awaits the requested historical billing reconciliation and a frozen execution plan."}


def bounded_luna_schedule(inventory, maximum_costs, available_usd, *, batch_size=5):
    """Prospective batches, never selecting by observed task success.

    Only absent cells are scheduled. The historical 404 cells are separate and
    require a later explicit recovery decision after billing reconciliation.
    Each batch must be reserved in full with 20% contingency before dispatch.
    Later batches are conditional on newly settled costs and refreshed catalogs.
    """
    if type(available_usd) not in (int, float) or not math.isfinite(available_usd) or available_usd < 0:
        raise ValueError("invalid remaining budget")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("invalid batch size")
    cells = inventory["missing"]
    ids = [cell["episode_id"] for cell in cells]
    if len(set(ids)) != len(ids) or set(maximum_costs) != set(ids):
        raise ValueError("cost estimates must match exactly the missing cells")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in maximum_costs.values()):
        raise ValueError("invalid conservative cost")
    seen_groups = set()
    for cell in cells:
        if cell["split"] not in {"train", "calibration", "threshold_selection"}:
            raise ValueError("schedule contains a non-development split")
        if cell["group_id"] in seen_groups:
            raise ValueError("multiple scheduled rows from the same independent group")
        seen_groups.add(cell["group_id"])
    ordered = []
    for split in ("train", "calibration", "threshold_selection"):
        families = sorted({c["family"] for c in cells if c["split"] == split})
        queues = {f: sorted((c for c in cells if c["split"] == split and c["family"] == f),
                            key=lambda c: digest(["luna-continuation-v1", c["episode_id"]])) for f in families}
        while any(queues.values()):
            ordered.extend(queues[f].pop(0) for f in families if queues[f])
    batches = []
    for split in ("train", "calibration", "threshold_selection"):
        subset = [c for c in ordered if c["split"] == split]
        for index in range(0, len(subset), batch_size):
            batch = subset[index:index + batch_size]
            maximum = sum(maximum_costs[c["episode_id"]] for c in batch)
            cap = math.ceil(maximum * 1.2 * 100) / 100
            batches.append({"index": len(batches) + 1, "split": split, "cells": batch,
                            "maximum_usd": maximum, "cap_usd": cap})
    return {"batches": batches, "available_usd_at_planning": available_usd,
            "first_batch_fits": bool(batches) and batches[0]["cap_usd"] <= available_usd,
            "all_batches_worst_case_usd": sum(b["cap_usd"] for b in batches),
            "collection_allowed": False,
            "selection_rule": "Original split order; round-robin family queues sorted by fixed identity digest; no outcome-based selection.",
            "remaining_batches_conditional": True}


def require_luna_billing_clearance(inventory, effective_outcomes, evidence_directory):
    """Reject until both historical outcomes have provider-evidenced charges.

    This gate does not infer a charge, amend a ledger, or accept dashboard
    absence as proof. Evidence itself must be verified before creating the
    canonical reconciliation; the future frozen plan binds those exact bytes.
    """
    values = [v for v in effective_outcomes if v.policy_id == LUNA]
    outcomes = {v.episode_id: v for v in values}
    if len(values) != len(outcomes):
        raise ValueError("Luna billing gate: duplicate outcomes")
    for cell in inventory["billing_blocked"]:
        value = outcomes.get(cell["episode_id"])
        if value is None or value.cost_usd is None:
            raise ValueError("Luna billing gate: historical 404 charge is unresolved")
        if type(value.cost_usd) not in (int, float) or not math.isfinite(value.cost_usd) or value.cost_usd < 0:
            raise ValueError("Luna billing gate: invalid reconciled charge")
        proof = value.evidence.get("provider_billing_reconciliation", {})
        expected = {a["attempt_id"]: a["terminal_hash"] for a in cell["attempts"]}
        if (proof.get("original_outcome_hash") != cell["original_outcome_hash"]
                or proof.get("attempt_terminal_hashes") != expected
                or proof.get("basis") not in {"provider_generation_metadata", "provider_explicit_billing_confirmation"}
                or not proof.get("provider_evidence_sha256")
                or not proof.get("provider_evidence_path")
                or proof.get("verified_cost_usd") != value.cost_usd):
            raise ValueError("Luna billing gate: provider-evidenced reconciliation is missing")
        root = Path(evidence_directory).resolve()
        relative = Path(proof["provider_evidence_path"])
        source = root / relative
        if (relative.is_absolute() or not source.resolve().is_relative_to(root)
                or source.is_symlink() or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != proof["provider_evidence_sha256"]):
            raise ValueError("Luna billing gate: provider evidence file is absent or changed")


def require_luna_batch_budget(batch, budget_summary):
    """Check full batch headroom against both development and Phase A caps."""
    available = min(14.46 - budget_summary["committed_usd"],
                    budget_summary["stages"]["development"]["remaining_usd"])
    if batch["cap_usd"] > available:
        raise ValueError("Luna batch requires more than the remaining authorized budget")


def verify_retained_history(parent, plan, authorization):
    """Unattempted cells may proceed while completed unknown charges stay reserved.

    This is not billing reconciliation or permission to repeat a historical
    cell. The canonical missing-cell schedule and shared claims enforce that.
    """
    import json
    if (authorization.get("collect_only_unattempted_with_history_reserved") is not True
            or not plan.get("historical_reservations")):
        raise ValueError("Unattempted continuation must explicitly retain historical reservations")
    for frozen in plan["historical_reservations"]:
        if frozen.get("stage") not in {"pilot", "development", "smoke", "test"}:
            raise ValueError("Invalid historical reservation stage")
        ledger = json.loads((Path(parent) / "budget" / (frozen["stage"] + ".json")).read_text())
        found = [e for e in ledger["entries"] if e.get("reservation_id") == frozen["reservation_id"]]
        if (len(found) != 1 or digest(found[0]) != frozen["entry_hash"]
                or found[0].get("actual_cost_usd") is not None
                or not found[0].get("terminal_unknown_evidence_hash")):
            raise ValueError("Historical unknown reservation was changed, released, or remains pending")


def verified_no_dispatch_inventory(parent, original, proofs):
    """Reconstruct eligible cells from immutable source ledgers, not grades."""
    import json
    from .episode_study import read_events
    from .episode_protocol import verify_protocol

    root = Path(parent).resolve()
    by_id = {r.episode_id: r for r in original}
    cells, prior_owners = [], {}
    required = {"execution-plan.json", "protocol.json", "development-outcomes.jsonl",
                "attempts.jsonl", "requests.jsonl"}
    for proof in proofs:
        relative = Path(proof["source_batch"])
        source = (root / relative).resolve()
        if relative.is_absolute() or source == root or not source.is_relative_to(root):
            raise ValueError("Invalid no-dispatch source directory")
        if set(proof["file_hashes"]) != required:
            raise ValueError("Incomplete no-dispatch evidence binding")
        for name, expected in proof["file_hashes"].items():
            file = source / name
            actual = hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
            if actual != expected or (expected is None and name != "requests.jsonl"):
                raise ValueError("No-dispatch source evidence changed")
        verify_protocol(json.loads((source / "protocol.json").read_text()), check_live=False)
        plan = json.loads((source / "execution-plan.json").read_text())
        row = by_id.get(proof["episode_id"])
        if (row is None or row.split not in {"train", "calibration", "threshold_selection"}
                or row.split != proof["split"] or row.content_hash != proof["original_row_hash"]
                or row.episode_id in prior_owners):
            raise ValueError("No-dispatch identity or original split mismatch")
        planned = [c for c in plan["batch"]["cells"] if c["episode_id"] == row.episode_id]
        if len(planned) != 1 or planned[0]["original_row_hash"] != row.content_hash:
            raise ValueError("No-dispatch source plan does not bind original row")
        outcomes = [e["outcome"] for e in read_events(source / "development-outcomes.jsonl")
                    if e.get("event") == "episode_completed" and e["outcome"]["episode_id"] == row.episode_id
                    and e["outcome"]["policy_id"] == LUNA]
        events = [e for e in read_events(source / "attempts.jsonl")
                  if e.get("episode_id") == row.episode_id and e.get("model") == LUNA]
        starts = [e["attempt_id"] for e in events if e.get("event") == "attempt_started"]
        terminals = [e for e in events if e.get("event") == "attempt_finished"]
        if (len(outcomes) != 1 or digest(outcomes[0]) != proof["original_outcome_hash"]
                or not terminals or len(starts) != len(set(starts))
                or len(terminals) != len(starts) or set(starts) != {e["attempt_id"] for e in terminals}
                or sorted(digest(e) for e in terminals) != sorted(proof["model_terminal_hashes"])
                or any(e.get("request_dispatched") is not False or e.get("cost_usd") != 0 for e in terminals)
                or any(e.get("episode_id") == row.episode_id for e in read_events(source / "requests.jsonl"))):
            raise ValueError("Cell lacks complete known-zero no-dispatch proof")
        cells.append({"episode_id": row.episode_id, "group_id": row.group_id,
                      "family": row.family, "split": row.split, "original_row_hash": row.content_hash})
        prior_owners[row.episode_id] = {"directory": str(source), "plan_hash": digest(plan)}
    if not cells:
        raise ValueError("Empty no-dispatch recovery")
    return {"missing": cells}, prior_owners


async def wait_for_openrouter_cooldown():
    """Honor the existing breaker without resetting it or creating attempts."""
    import asyncio
    from theriad.runtime.circuit_breaker import get_circuit_registry
    while True:
        states = get_circuit_registry().snapshot("llm:openrouter")
        if any(s.get("half_open_in_flight") for s in states):
            raise RuntimeError("OpenRouter half-open request still in flight")
        delays = [s["retry_after_seconds"] for s in states if s["status"] == "open"]
        if not delays:
            return
        await asyncio.sleep(min(30, max(delays) + .05))


class LunaCollectionPaused(RuntimeError):
    """A provider rate limit or local rejection needs a separate continuation."""


async def run_missing_luna_batch(parent, directory):
    """Run one explicitly frozen batch through the existing episode collector.

    A prepared schedule is not executable authorization. Historical charges
    must be reconciled or explicitly retained in full for unattempted cells.
    No model fitting, split reassignment, or automatic next batch is performed.
    """
    from dataclasses import replace
    from .episode_recovery import study_cli
    from .episode_protocol import verify_protocol
    from .episode_engineering import require_paid_authorization, verify_admission
    from .episode_runtime import EpisodeRuntime
    from .episode_diagnostics import diagnostic_provider
    from .episode_study import ProductStudyBudget, _collector_lock, read_events, atomic_json, append_event
    from theriad.core.providers.openrouter_capabilities import SNAPSHOT_KEY, compatible_payload

    parent, directory = Path(parent).resolve(), Path(directory).resolve()
    if directory == parent or not directory.is_relative_to(parent):
        raise ValueError("Luna batch needs a separate child directory")
    cli = study_cli()
    plan = cli.read_json(directory / "execution-plan.json")
    protocol = cli.read_json(directory / "protocol.json")
    authorization = cli.read_json(directory / "authorization.json")
    verify_protocol(protocol)
    if (authorization.get("approved") is not True
            or not authorization.get("user_authorization_reference")
            or authorization.get("plan_hash") != digest(plan)
            or authorization.get("protocol_hash") != protocol["manifest_hash"]):
        raise ValueError("Luna batch authorization does not bind the execution plan and source")
    contract, _, controls = cli.validate_contract(parent, cli.DEVELOPMENT_SPLITS)
    require_paid_authorization(parent, contract, phase="development")
    if (plan.get("parent_contract_hash") != digest(contract)
            or plan.get("parent_outcomes_hash") != digest(read_events(parent / "development-outcomes.jsonl"))
            or plan.get("parent_attempts_hash") != digest(read_events(parent / "attempts.jsonl"))):
        raise ValueError("Luna batch foundation changed")
    original = cli.load_rows(parent, cli.DEVELOPMENT_SPLITS)
    raw_outcomes = [cli.EpisodeOutcome(**e["outcome"]) for e in read_events(parent / "development-outcomes.jsonl")
                    if e.get("event") == "episode_completed"]
    inventory = missing_luna_cells(original, raw_outcomes, read_events(parent / "attempts.jsonl"))
    # Evaluate this before creating runtime/provider objects or changing ledgers.
    if authorization.get("collect_only_unattempted_with_history_reserved"):
        verify_retained_history(parent, plan, authorization)
    else:
        require_luna_billing_clearance(inventory, cli.outcomes(parent / "development-outcomes.jsonl"), parent)
    if len(inventory["billing_blocked"]) != 2:
        raise ValueError("Expected the two frozen historical Luna 404s")
    if plan.get("max_output_tokens") != 128000 or plan.get("max_seconds") != 600:
        raise ValueError("Luna execution limits changed")
    if set(plan.get("metadata", {})) != {LUNA}:
        raise ValueError("Luna batch cannot collect other policies")
    quote = plan["metadata"][LUNA]
    compatible_payload({"model": LUNA, "messages": [], "temperature": 0, "max_tokens": 128000,
                        "tools": [], "response_format": {"type": "json_object"}},
                       {SNAPSHOT_KEY: quote.get(SNAPSHOT_KEY)})
    if quote.get("provider_only") != "openai/flex" or quote[SNAPSHOT_KEY]["endpoint_tag"] != "openai/flex":
        raise ValueError("Luna batch must use the reviewed exact endpoint")
    runtime = EpisodeRuntime(directory=directory, metadata=plan["metadata"], baseline_plans=controls,
        provider_factory=lambda provider, **kw: diagnostic_provider(directory, provider, 600),
        use_full_output_limit=True, archive_responses=True)
    runtime.study_protocol = protocol
    policy = {"kind": "model", "model": LUNA}
    prior_owners = {}
    if plan.get("version") == "luna-no-dispatch-recovery-v1":
        if authorization.get("recover_proven_no_dispatch_only") is not True:
            raise ValueError("No-dispatch recovery authorization missing")
        inventory, prior_owners = verified_no_dispatch_inventory(parent, original, plan["no_dispatch_proofs"])
    missing_ids = {c["episode_id"] for c in inventory["missing"]}
    derived = {r.episode_id: replace(r, max_output_tokens=128000, max_seconds=600)
               for r in original if r.episode_id in missing_ids}
    costs = {key: runtime.maximum_cost(row, policy) for key, row in derived.items()}
    schedule = bounded_luna_schedule(inventory, costs, 0)
    index = plan.get("batch_index")
    if type(index) is not int or not 1 <= index <= len(schedule["batches"]):
        raise ValueError("Invalid prospective Luna batch index")
    batch = schedule["batches"][index - 1]
    if plan.get("batch") != batch:
        raise ValueError("Luna batch differs from the prospective missing-cell schedule")
    rows = [derived[c["episode_id"]] for c in batch["cells"]]
    verify_admission(parent, [r for r in original if r.episode_id in {c["episode_id"] for c in batch["cells"]}], contract)
    budget = ProductStudyBudget(parent / "budget")
    if budget.stage_caps["pilot"] + budget.stage_caps["development"] > 14.46:
        raise ValueError("Original Phase A cap changed")
    with _collector_lock(budget):
        # The gate is repeated under the collection lock. Parent mutations are
        # prohibited during a batch; future batches freeze their own bindings.
        if (plan["parent_outcomes_hash"] != digest(read_events(parent / "development-outcomes.jsonl"))
                or plan["parent_attempts_hash"] != digest(read_events(parent / "attempts.jsonl"))):
            raise ValueError("Luna foundation changed while acquiring the collector lock")
        if authorization.get("collect_only_unattempted_with_history_reserved"):
            verify_retained_history(parent, plan, authorization)
        if not read_events(directory / "development-outcomes.jsonl"):
            require_luna_batch_budget(batch, budget.summary())
        claims_path = parent / "luna-compatibility-claims.jsonl"
        claims = read_events(claims_path)
        owner = {"directory": str(directory), "plan_hash": digest(plan)}
        for row in rows:
            existing = [c for c in claims if c.get("episode_id") == row.episode_id]
            allowed = [owner]
            if row.episode_id in prior_owners:
                allowed.append(prior_owners[row.episode_id])
                if not any(all(c.get(k) == v for k, v in allowed[1].items()) for c in existing):
                    raise ValueError("Original no-dispatch claim missing")
            if (len(existing) > len(allowed) or len({(c.get("directory"), c.get("plan_hash")) for c in existing}) != len(existing) or any(
                    not any(all(c.get(k) == v for k, v in a.items()) for a in allowed) for c in existing)):
                raise ValueError("Luna cell already belongs to another batch; reconcile instead of replaying")
        cli.preflight(directory)
        for row in rows:
            if not any(c.get("episode_id") == row.episode_id and
                       all(c.get(k) == v for k, v in owner.items()) for c in claims):
                append_event(claims_path, {"episode_id": row.episode_id, "policy_id": LUNA, **owner})
        for row in rows:
            await wait_for_openrouter_cooldown()
            before = len(read_events(directory / "attempts.jsonl"))
            await cli._collect_row(row, {LUNA: policy}, stage="development", runtime=runtime,
                directory=directory, contract={"profile": "luna-compatibility", "plan_hash": digest(plan),
                                                "protocol_hash": protocol["manifest_hash"]},
                budget_override=budget, allow_bounded_unknown=True)
            fresh = read_events(directory / "attempts.jsonl")[before:]
            if any(e.get("event") == "attempt_finished" and
                   (e.get("provider_status_code") == 429 or e.get("error_type") == "CircuitOpenError")
                   for e in fresh):
                atomic_json(directory / "pause.json", {"reason": "rate_limit_or_local_circuit_rejection",
                    "episode_id": row.episode_id, "automatic_retry": False})
                raise LunaCollectionPaused("Collection paused after rate limit or local circuit rejection")
        values = cli.outcomes(directory / "development-outcomes.jsonl")
        atomic_json(directory / "report.json", {"observations": [asdict(v) for v in values],
            "status": "completed" if len(values) == len(rows) else "incomplete",
            "original_split": batch["split"], "fitting_permitted": False,
            "promotion_eligible": False, "parent_budget": budget.summary()})
