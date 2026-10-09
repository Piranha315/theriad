#!/usr/bin/env python3
"""Reproducible, opt-in product episode study; never activates production routing.

Run ``prepare`` (read-only provider catalog), then ``run`` (authorized paid
collection), or ``report`` (offline). Every paid dispatch follows a durable
reservation. A changed study contract requires a new, separately budgeted study;
this command never silently resets an existing ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from theriad.routing.episode_study import (  # noqa: E402
    FAMILIES,
    VERSION,
    Episode,
    EpisodeOutcome,
    ProductStudyBudget,
    append_event,
    atomic_json,
    digest,
    estimate_power,
    execute_pair,
    freeze_test,
    paired_report,
    read_events,
    validate_episodes,
    verify_lock,
)
from theriad.routing.study_budget import StudyBudgetExceeded  # noqa: E402

DEFAULT_ALTERNATIVES = ("deepseek/deepseek-v4-flash-0731", "openai/gpt-5.6-luna")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, payload):
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"refusing to replace frozen {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def load_rows(directory, splits):
    # Selection only opens development files. Test data is opened after the
    # policy artifact exists and is immutable.
    contract = read_json(directory / "contract.json")
    rows = []
    for split in splits:
        values = read_json(directory / "fixtures" / f"{split}.json")
        if digest(values) != contract["split_hashes"][split]:
            raise ValueError("frozen split content changed")
        rows.extend(Episode(**value) for value in values)
    return rows


def outcomes(path):
    return effective_outcomes(read_events(path))


def effective_outcomes(events):
    values = []
    for event in events:
        if event.get("event") == "episode_completed":
            values.append(event["outcome"])
        elif event.get("event") == "episode_cost_reconciled":
            matching = [i for i, value in enumerate(values) if digest(value) == event["original_outcome_hash"]]
            if len(matching) != 1:
                raise ValueError("cost reconciliation does not match one durable outcome")
            before, after = values[matching[0]], event["outcome"]
            if before["cost_usd"] is not None or {k: v for k, v in before.items() if k not in {"cost_usd", "evidence"}} != {k: v for k, v in after.items() if k not in {"cost_usd", "evidence"}}:
                raise ValueError("cost reconciliation changed identity or quality")
            values[matching[0]] = after
    return [EpisodeOutcome(**value) for value in values]


def effective_completions(events):
    terminals = []
    for event in events:
        if event.get("event") == "collection_completed":
            terminals.append(event)
        elif event.get("event") == "collection_cost_reconciled":
            matching = [i for i, terminal in enumerate(terminals) if digest(terminal) == event["original_completion_hash"]]
            if len(matching) != 1:
                raise ValueError("collection cost reconciliation has no unique original")
            old, new = terminals[matching[0]], event["completion"]
            if old["cost_usd"] is not None or new["cost_usd"] is None or any(old.get(key) != new.get(key) for key in ("event", "collection_id", "reservation_id")):
                raise ValueError("collection cost reconciliation changed identity")
            terminals[matching[0]] = new
    return terminals


def reconcile_free_failures(directory):
    """Reconcile a provably zero-price failed baseline request, never a paid one."""
    from dataclasses import replace

    from theriad.routing.episode_hooks import zero_charge_failure_bound
    from theriad.routing.episode_study import _collector_lock
    contract, catalog, plans = validate_contract(directory)
    budget = ProductStudyBudget(directory / "budget")
    count = 0
    with _collector_lock(budget):
        for stage in ("pilot", "development"):
            path = directory / f"{stage}-outcomes.jsonl"
            for original in outcomes(path):
                if original.cost_usd is not None or original.policy_id != "baseline":
                    continue
                plan = plans[original.episode_id]
                if not plan.get("chain"):
                    continue
                attempts = [e for e in read_events(directory / "attempts.jsonl") if e.get("episode_id") == original.episode_id and e.get("policy_id") == original.policy_id]
                starts = {e["attempt_id"] for e in attempts if e.get("event") == "attempt_started"}
                terminals = [e for e in attempts if e.get("event") == "attempt_finished"]
                if len(terminals) != len(starts) or {e["attempt_id"] for e in terminals} != starts:
                    continue
                unknown = [e for e in terminals if e.get("cost_usd") is None]
                if not unknown:
                    continue
                for event in unknown:
                    model = event.get("model", "")
                    extra = plan.get("request_extras", {}).get(model)
                    if (event.get("kind") != "model" or event.get("status") != "failed" or event.get("role") != "completion" or extra is None or [event.get("provider"), model] not in plan["chain"] or zero_charge_failure_bound(model, event.get("provider", ""), catalog.get(model, {}), extra) is None):
                        break
                else:
                    evidence = {"basis": "frozen_zero_charge_upper_bound", "billed_cost_observed": False,
                        "attempt_ids": [e["attempt_id"] for e in unknown], "catalog_hash": contract["catalog_hash"],
                        "control_hash": digest(plan), "known_total_cost_usd": sum(e["cost_usd"] for e in terminals if e["cost_usd"] is not None)}
                    amended = replace(original, cost_usd=evidence["known_total_cost_usd"], evidence={**original.evidence, "cost_reconciliation": evidence})
                    source = next(e for e in read_events(path) if e.get("event") == "episode_completed" and e["outcome"]["episode_id"] == original.episode_id and e["outcome"]["policy_id"] == original.policy_id)
                    append_event(path, {"event": "episode_cost_reconciled", "collection_id": source["collection_id"], "original_outcome_hash": digest(asdict(original)), "outcome": asdict(amended)})
                    count += 1
            events = read_events(path)
            for start in [e for e in events if e.get("event") == "collection_started"]:
                group = [e for e in events if e.get("collection_id") == start["collection_id"]]
                completed = effective_completions(group)
                if len(completed) > 1:
                    raise ValueError("duplicate collection completion")
                saved = effective_outcomes(group)
                if len(saved) == len(start["configurations"]) and {r.policy_id for r in saved} == set(start["configurations"]) and all(r.cost_usd is not None for r in saved):
                    total = sum(r.cost_usd for r in saved)
                    terminal = {"event": "collection_completed", "collection_id": start["collection_id"], "reservation_id": start["reservation_id"], "cost_usd": total, "reconciled": True}
                    if completed and completed[0]["cost_usd"] is None:
                        append_event(path, {"event": "collection_cost_reconciled", "collection_id": start["collection_id"],
                            "original_completion_hash": digest(completed[0]), "completion": terminal})
                    elif not completed:
                        append_event(path, terminal)
                    elif not math.isclose(completed[0]["cost_usd"], total, abs_tol=1e-12):
                        raise ValueError("reconciliation changed a known collection total")
                    budget.settle(stage, start["reservation_id"], total)
    return count


def eligible(row, model, catalog):
    from theriad.routing.episode_runtime import derived_capabilities

    quote = catalog.get(model)
    return bool(
        quote
        and set(row.capabilities) <= derived_capabilities(quote)
        and quote["context_window"] >= row.max_input_tokens + row.max_output_tokens
    )


def load_credentials():
    # Existing local provider configuration only. Never serialize environment,
    # keys, response headers, or provider exception messages into artifacts.
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from theriad.settings.secrets import get_provider_api_key

    if not get_provider_api_key("openrouter"):
        raise RuntimeError("OpenRouter credential unavailable in local configuration")


async def prepare(directory, alternative_models, corpus_path=None):
    import httpx

    from theriad.routing.episode_fixtures import build_seed_corpus, corpus_manifest
    from theriad.routing.episode_runtime import freeze_baseline

    if (directory / "contract.json").exists():
        validate_contract(directory)
        return
    rows = build_seed_corpus() if corpus_path is None else [Episode(**json.loads(line)) for line in Path(corpus_path).read_text().splitlines() if line.strip()]
    validate_episodes(rows)
    for split in ("pilot", "train", "validation", "test", "smoke"):
        if {row.family for row in rows if row.split == split} != set(FAMILIES):
            raise ValueError(f"all five workload families are required in split {split}")
    plans = {row.episode_id: freeze_baseline(row) for row in rows}
    required = set(alternative_models)
    for plan in plans.values():
        required.update(model for _, model in plan.get("chain", []))
        required.update(plan.get("models", {}).values())
    async with httpx.AsyncClient(timeout=40) as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        live = {item["id"]: item for item in response.json()["data"]}
    catalog = {}
    for model in sorted(required):
        if model not in live:
            raise ValueError(f"frozen baseline/alternative absent from provider catalog: {model}")
        raw = live[model]
        price = raw.get("pricing", {})
        if price.get("prompt") is None or price.get("completion") is None:
            raise ValueError(f"missing price for {model}")
        input_price, output_price = float(price["prompt"]), float(price["completion"])
        if not all(math.isfinite(x) and x >= 0 for x in (input_price, output_price)):
            raise ValueError("invalid provider price")
        modalities = raw.get("architecture", {}).get("input_modalities", [])
        capabilities = ["text", "search"]
        if "image" in modalities:
            capabilities.append("vision")
        if "tools" in raw.get("supported_parameters", []):
            capabilities.append("tools")
        catalog[model] = {
            "provider": "openrouter",
            "input_per_1m": input_price * 1e6,
            "output_per_1m": output_price * 1e6,
            "context_window": raw.get("context_length", 0),
            "capabilities": capabilities,
            "request_cost_usd": float(price.get("request", 0)),
            "image_cost_usd": float(price.get("image", 0)),
            "provider_pricing": price,
        }
    for split in ("pilot", "train", "validation", "test", "smoke"):
        write_once(
            directory / "fixtures" / f"{split}.json",
            [asdict(row) for row in rows if row.split == split],
        )
        write_once(
            directory / "controls" / f"{split}.json",
            {row.episode_id: plans[row.episode_id] for row in rows if row.split == split},
        )
    write_once(directory / "catalog.json", catalog)
    manifest = corpus_manifest(rows) if corpus_path is None else {**validate_episodes(rows),
        "fixture_versions": sorted({row.fixture_version for row in rows}),
        "grader_versions": sorted({row.grader_version for row in rows}),
        "sources": sorted({row.source for row in rows}), "source_kind": "provided_versioned_corpus",
        "power_assessed": False, "live_route_evidence": False}
    manifest["independent_groups_by_family_split"] = {family: {split: len({r.group_id for r in rows if r.family == family and r.split == split})
        for split in ("pilot", "train", "validation", "test", "smoke")} for family in FAMILIES}
    write_once(directory / "corpus-manifest.json", manifest)
    contract = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_hash": digest(catalog),
        "baseline_hash": digest(plans),
        "split_hashes": {
            split: digest([asdict(r) for r in rows if r.split == split])
            for split in set(r.split for r in rows)
        },
        "control_hashes": {
            split: digest({r.episode_id: plans[r.episode_id] for r in rows if r.split == split})
            for split in set(r.split for r in rows)
        },
        "corpus_hash": digest([asdict(row) for row in rows]),
        "alternatives": list(alternative_models),
        "seed": 731,
        "budget_usd": 100,
        "selection_cost_ratio": 0.95,
        "family_weights": {family: 0.2 for family in FAMILIES},
        "scope": "resettable product fixtures; no customer data or deployment",
        "corpus_source": "built_in_seed" if corpus_path is None else "provided_versioned_corpus",
        "judge": "versioned deterministic checks; no paid judge or embedding calls",
    }
    write_once(directory / "contract.json", contract)
    ProductStudyBudget(directory / "budget").summary()


def validate_contract(directory, splits=("pilot", "train", "validation")):
    contract = read_json(directory / "contract.json")
    catalog = read_json(directory / "catalog.json")
    plans = {}
    for split in splits:
        values = read_json(directory / "controls" / f"{split}.json")
        if digest(values) != contract["control_hashes"][split]:
            raise ValueError("frozen control changed")
        plans.update(values)
    if digest(catalog) != contract["catalog_hash"]:
        raise ValueError("frozen catalog/control changed")
    return contract, catalog, plans


def preflight(directory):
    from theriad.routing.sandbox import CODE_GRADER_RUNTIME_ENV, run_python_tests

    os.environ.setdefault(CODE_GRADER_RUNTIME_ENV, str(ROOT / ".venv" / "bin" / "python"))
    result = run_python_tests("def add(a,b): return a+b", "assert add(2,3)==5", entry_point="add")
    evidence = {
        "code_sandbox": result.to_dict(),
        "provider_key_present": True,
        "real_browser_backend_verified": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(directory / "preflight.json", evidence)
    if not result.passed:
        raise RuntimeError("isolated executable grading preflight failed; no paid episodes started")


async def collect_row(row, configurations, *, stage, runtime, directory, contract, allow_bounded_unknown=False):
    from theriad.routing.episode_study import _collector_lock

    with _collector_lock(ProductStudyBudget(directory / "budget")):
        return await _collect_row(
            row,
            configurations,
            stage=stage,
            runtime=runtime,
            directory=directory,
            contract=contract,
            allow_bounded_unknown=allow_bounded_unknown,
        )


async def _collect_row(row, configurations, *, stage, runtime, directory, contract, allow_bounded_unknown=False):
    path = directory / f"{stage}-outcomes.jsonl"
    identity = digest([digest(contract), stage, row.content_hash, configurations])
    events = [event for event in read_events(path) if event.get("collection_id") == identity]
    completed = effective_completions(events)
    budget = ProductStudyBudget(directory / "budget")
    if events and not completed and allow_bounded_unknown:
        saved = effective_outcomes(events)
        starts = [e for e in events if e.get("event") == "collection_started"]
        if len(starts) == 1 and len(saved) == len(configurations) and {r.policy_id for r in saved} == set(configurations) and not any(r.evidence.get("accounting", {}).get("pending_ids") for r in saved):
            total = sum(r.cost_usd for r in saved) if all(r.cost_usd is not None for r in saved) else None
            terminal = {"event": "collection_completed", "collection_id": identity,
                "reservation_id": starts[0]["reservation_id"], "cost_usd": total, "recovered_terminal_outcomes": True}
            append_event(path, terminal)
            completed = [terminal]
    if completed:
        if len(completed) != 1:
            raise ValueError("duplicate collection completion")
        saved = [asdict(result) for result in effective_outcomes(events)]
        if (
            len(saved) != len(configurations)
            or {r["policy_id"] for r in saved} != set(configurations)
            or (not allow_bounded_unknown and any(r["cost_usd"] is None for r in saved))
        ):
            raise ValueError("incomplete durable configuration observations")
        total = sum(r["cost_usd"] for r in saved) if all(r["cost_usd"] is not None for r in saved) else None
        if total != completed[0]["cost_usd"]:
            raise ValueError("configuration settlement does not match observations")
        if total is None and allow_bounded_unknown:
            budget.retain_terminal_unknown(stage, completed[0]["reservation_id"], digest(saved), known_incurred_cost_usd=sum(r["cost_usd"] if r["cost_usd"] is not None else r.get("evidence", {}).get("accounting", {}).get("known_total_cost_usd", 0) for r in saved))
        else:
            budget.settle(stage, completed[0]["reservation_id"], total)
        return
    if events:
        raise StudyBudgetExceeded("interrupted collection retained; reconcile before retry")
    maximum = sum(runtime.maximum_cost(row, policy) for policy in configurations.values())
    reservation = budget.reserve(
        stage, identity, maximum, calls=row.max_calls * len(configurations)
    )
    append_event(
        path,
        {
            "event": "collection_started",
            "collection_id": identity,
            "episode_id": row.episode_id,
            "configurations": configurations,
            "reservation_id": reservation,
            "maximum_cost_usd": maximum,
        },
    )
    order = sorted(configurations)
    random.Random(identity).shuffle(order)
    collected = []
    for policy_id in order:
        result = await asyncio.wait_for(
            runtime(row, policy_id, configurations[policy_id]), row.max_seconds
        )
        if (
            result.episode_id,
            result.group_id,
            result.family,
            result.policy_id,
            result.fixture_hash,
        ) != (row.episode_id, row.group_id, row.family, policy_id, row.content_hash):
            raise ValueError("episode adapter identity mismatch")
        append_event(
            path,
            {"event": "episode_completed", "collection_id": identity, "outcome": asdict(result)},
        )
        collected.append(asdict(result))
        print(
            json.dumps(
                {
                    "stage": stage,
                    "episode": row.episode_id,
                    "policy": policy_id,
                    "success": result.success,
                    "status": result.status,
                    "cost_usd": result.cost_usd,
                }
            ),
            flush=True,
        )
        if result.cost_usd is None and (not allow_bounded_unknown or result.evidence.get("accounting", {}).get("pending_ids")):
            raise StudyBudgetExceeded("unknown charge retained; collection stopped")
    total = sum(r["cost_usd"] for r in collected) if all(r["cost_usd"] is not None for r in collected) else None
    append_event(
        path,
        {
            "event": "collection_completed",
            "collection_id": identity,
            "reservation_id": reservation,
            "cost_usd": total,
        },
    )
    if total is None and allow_bounded_unknown:
        budget.retain_terminal_unknown(stage, reservation, digest(collected), known_incurred_cost_usd=sum(r["cost_usd"] if r["cost_usd"] is not None else r.get("evidence", {}).get("accounting", {}).get("known_total_cost_usd", 0) for r in collected))
    else:
        budget.settle(stage, reservation, total)


def configuration_score(rows, observed, choices):
    if len({(result.episode_id, result.policy_id) for result in observed}) != len(observed):
        raise ValueError("duplicate policy outcomes")
    by_key = {(result.episode_id, result.policy_id): result for result in observed}
    quality, cost = [], []
    for family in FAMILIES:
        members = [row for row in rows if row.family == family]
        selected = [by_key.get((row.episode_id, choices[row.episode_id])) for row in members]
        if not members or any(
            result is None or result.success is None or result.cost_usd is None
            for result in selected
        ):
            return None
        if any(
            (result.fixture_hash, result.group_id, result.family)
            != (row.content_hash, row.group_id, row.family)
            for row, result in zip(members, selected, strict=True)
        ):
            raise ValueError("observed selection outcome does not match its frozen fixture")
        quality.append(sum(result.success for result in selected) / len(selected))
        cost.append(sum(result.cost_usd for result in selected) / len(selected))
    return {"success": sum(quality) / 5, "cost_usd": sum(cost) / 5}


def select_challenger(directory, runtime, contract, catalog):
    # This function deliberately never opens test/smoke rows or outcomes.
    rows = load_rows(directory, ["train", "validation"])
    validation = [row for row in rows if row.split == "validation"]
    observed = outcomes(directory / "development-outcomes.jsonl")
    if len({(o.episode_id, o.policy_id) for o in observed}) != len(observed):
        raise ValueError("duplicate development outcomes")
    base = configuration_score(validation, observed, {r.episode_id: "baseline" for r in validation})
    if base is None:
        raise ValueError("complete baseline validation observations required")
    options = {}
    for family in FAMILIES:
        members = [row for row in rows if row.family == family]
        options[family] = ["baseline"] + [
            model
            for model in contract["alternatives"]
            if all(eligible(row, model, catalog) for row in members)
        ]
    comparisons = []
    for selected in itertools.product(*(options[family] for family in FAMILIES)):
        family_choice = dict(zip(FAMILIES, selected, strict=True))
        choices = {row.episode_id: family_choice[row.family] for row in validation}
        score = configuration_score(validation, observed, choices)
        if score is None:
            continue
        policy = {
            "kind": "by_family",
            "policies": {
                family: {"kind": "current"}
                if model == "baseline"
                else {"kind": "model", "model": model}
                for family, model in family_choice.items()
            },
        }
        comparisons.append(
            {"policy": policy, **score, "complexity": len(set(selected)), "family": "task_fixed"}
        )
    # Fit existing surface predictors from complete episode outcomes. Failure
    # to meet training support is reported; test data never fills missing cells.
    learner_diagnostics = []
    try:
        from theriad.models.tiers import ModelCandidate
        from theriad.routing.v2_training import ModelOutcome, StudyRow, train_surface_artifact

        indexed = {(o.episode_id, o.policy_id): o for o in observed}
        training_rows = []
        for row in rows:
            cells = {}
            for model in contract["alternatives"]:
                value = indexed.get((row.episode_id, model))
                if value and value.success is not None and value.cost_usd is not None:
                    if value.evidence.get("output_tokens") is None:
                        raise ValueError(
                            "complete observed output tokens required for cost fitting"
                        )
                    cells[model] = ModelOutcome(
                        float(value.success),
                        value.cost_usd,
                        int(value.evidence["output_tokens"]),
                        value.latency_ms,
                    )
            authored = row.source in {"newly_authored", "public_reference_with_authored_question"}
            provenance = row.fixture.get("source_metadata", {})
            training_rows.append(
                StudyRow(
                    prompt=row.prompt,
                    split=row.split,
                    source=row.source,
                    benchmark_family=row.family,
                    domain=row.family,
                    template_family=row.template_id,
                    collected_at=contract["created_at"],
                    collection_vintage=contract["created_at"],
                    source_revision=row.fixture_version,
                    license_name=provenance.get("license_name", "user-authorized newly authored study tasks" if authored else ""),
                    license_status=provenance.get("license_status", "approved" if authored else ""),
                    provenance_url=provenance.get("provenance_url", f"repo:theriad/routing/episode_fixtures.py#{row.template_id}" if authored else ""),
                    lane="code" if row.family == "coding" else "general",
                    outcomes=cells,
                    prompt_id=row.episode_id,
                    leakage_group=row.group_id,
                    grader=row.grader,
                )
            )
        candidates = [
            ModelCandidate(
                tier="balanced",
                provider="openrouter",
                model=model,
                input_cost_per_million=catalog[model]["input_per_1m"],
                output_cost_per_million=catalog[model]["output_per_1m"],
                context_window=catalog[model]["context_window"],
            )
            for model in contract["alternatives"]
        ]
        artifact = train_surface_artifact(
            training_rows,
            fallback_models={lane: contract["alternatives"][0] for lane in ("general", "code")},
            active_models_by_lane={lane: list(contract["alternatives"]) for lane in ("general", "code")},
            model_candidates=candidates,
        )
        artifact["research_only"] = True
        atomic_json(directory / "surface-candidate.json", artifact)
        policy = {"kind": "surface", "artifacts": {family: artifact for family in FAMILIES}}
        choices = {row.episode_id: runtime.models_for(row, policy)[0] for row in validation}
        score = configuration_score(validation, observed, choices)
        if score is not None:
            comparisons.append({"policy": policy, **score, "complexity": 100, "family": "surface"})
        learner_diagnostics.append({"family": "surface", "status": "fitted", "score": score})
    except (ValueError, KeyError, RuntimeError) as error:
        learner_diagnostics.append(
            {"family": "surface", "status": "ineligible", "reason": str(error)}
        )
    learner_diagnostics.append(
        {
            "family": "hybrid",
            "status": "ineligible",
            "reason": "No frozen locally available embedding runtime; no download or paid embeddings used.",
        }
    )
    feasible = [
        candidate for candidate in comparisons if candidate["cost_usd"] <= 0.95 * base["cost_usd"]
    ]
    feasible.sort(
        key=lambda candidate: (
            -candidate["success"],
            candidate["cost_usd"],
            candidate["complexity"],
            digest(candidate["policy"]),
        )
    )
    summary = {
        "baseline": base,
        "cost_ceiling_usd": 0.95 * base["cost_usd"],
        "candidate_count": len(comparisons),
        "feasible_count": len(feasible),
        "learners": learner_diagnostics,
        "selection_uses_test": False,
        "candidates": [{k: v for k, v in item.items() if k != "policy"} for item in comparisons],
    }
    atomic_json(directory / "validation-selection.json", summary)
    if not feasible:
        raise RuntimeError(
            "no challenger meets validation cost headroom with complete observations"
        )
    selected = feasible[0]
    artifact = {
        **selected["policy"],
        "research_only": True,
        "production_enabled": False,
        "version": VERSION,
        "contract_hash": digest(contract),
        "validation": {k: v for k, v in selected.items() if k != "policy"},
    }
    write_once(directory / "challenger.json", artifact)
    return artifact


async def run_study(directory, *, stop_after):
    from theriad.routing.episode_protocol import capture_protocol, verify_protocol
    protocol = capture_protocol(directory)
    from theriad.routing.episode_runtime import EpisodeRuntime

    load_credentials()
    contract, catalog, plans = validate_contract(directory)
    preflight(directory)
    source_paths = ["scripts/run_router_episode_study.py", "theriad/routing/episode_runtime.py",
                    "theriad/routing/episode_study.py", "theriad/routing/episode_hooks.py",
                    "theriad/agents/model_telemetry.py", "theriad/benchmark/types.py"]
    import hashlib
    append_event(directory / "execution-history.jsonl", {"event": "execution_started",
        "at": datetime.now(timezone.utc).isoformat(), "contract_hash": digest(contract),
        "sources_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}})
    runtime = EpisodeRuntime(directory=directory, metadata=catalog, baseline_plans=plans)
    for stage, splits in (("pilot", ["pilot"]), ("development", ["train", "validation"])):
        rows = load_rows(directory, splits)
        random.Random(contract["seed"]).shuffle(rows)
        for row in rows:
            configurations = {"baseline": {"kind": "current"}}
            candidates = [m for m in contract["alternatives"] if eligible(row, m, catalog)]
            for model in candidates[: 1 if stage == "pilot" else 2]:
                configurations["challenger" if stage == "pilot" else model] = {
                    "kind": "model",
                    "model": model,
                }
            if stage == "pilot" and len(configurations) != 2:
                raise ValueError("pilot needs an eligible paired alternative")
            await collect_row(
                row,
                configurations,
                stage=stage,
                runtime=runtime,
                directory=directory,
                contract=contract,
                allow_bounded_unknown=True,
            )
        render_report(directory)
        if stop_after == stage:
            return
    if (directory / "challenger.json").exists():
        challenger = read_json(directory / "challenger.json")
        if challenger.get("contract_hash") != digest(contract):
            raise ValueError("frozen challenger belongs to a different study contract")
    else:
        challenger = select_challenger(directory, runtime, contract, catalog)
    pilot = outcomes(directory / "pilot-outcomes.jsonl")
    power = estimate_power(pilot=pilot)
    tests = load_rows(directory, ["test"])
    _, _, test_plans = validate_contract(directory, ["test"])
    runtime.baseline_plans.update(test_plans)
    # Independent seed groups are the limiting resource. Never manufacture
    # sample size by treating repeated templates as new independent tasks.
    power.update(
        {
            "available_independent_test_groups": len({r.group_id for r in tests}),
            "frozen_sample_reason": "all independent groups in the versioned seed corpus",
            "repeated_templates_increase_independent_n": False,
            "available_sample_supports_target": len({r.group_id for r in tests}) >= power["required_independent_groups"],
        }
    )
    policies = {"baseline": {"kind": "current"}, "challenger": challenger}
    verify_protocol(protocol)
    if (directory / "test-lock.json").exists():
        existing_lock = read_json(directory / "test-lock.json")
        frozen_protocol = existing_lock.get("protocol")
        if frozen_protocol is None or frozen_protocol.get("manifest_hash") != protocol["manifest_hash"]:
            raise ValueError("locked test has no matching verified source protocol; do not mix or replay paid test episodes")
    maximum = sum(
        runtime.maximum_cost(row, policy) for row in tests for policy in policies.values()
    )
    if maximum > 50:
        # Balanced deterministic budget reduction occurs before any test outcome.
        balanced = []
        for index in range(min(sum(r.family == family for r in tests) for family in FAMILIES)):
            batch = [
                sorted([r for r in tests if r.family == family], key=lambda x: x.episode_id)[index]
                for family in FAMILIES
            ]
            if (
                sum(runtime.maximum_cost(r, p) for r in balanced + batch for p in policies.values())
                > 50
            ):
                break
            balanced.extend(batch)
        tests = balanced
    independent_n = len({row.group_id for row in tests})
    import statistics
    z = statistics.NormalDist().inv_cdf(.95) + statistics.NormalDist().inv_cdf(.8)
    power.update({"frozen_independent_groups": independent_n,
        "approximate_detectable_effect_at_frozen_n": z * math.sqrt(power["used_variance"] / independent_n) if independent_n else None})
    write_once(directory / "power-plan.json", power)
    lock = freeze_test(
        directory / "test-lock.json",
        episodes=tests,
        baseline=policies["baseline"],
        challenger=challenger,
        pricing=catalog,
        power=power,
        seed=contract["seed"],
        protocol=protocol,
    )
    verify_lock(lock, tests)
    if stop_after == "selection":
        render_report(directory)
        return
    by_id = {row.episode_id: row for row in tests}
    async def verified_runner(row, policy_id, policy):
        from dataclasses import replace
        verify_protocol(protocol)
        value = await runtime(row, policy_id, policy)
        evidence = {**value.evidence, "protocol_manifest_hash": protocol["manifest_hash"]}
        try:
            verify_protocol(protocol)
        except (ValueError, OSError):
            evidence["protocol_verification_failed"] = True
        return replace(value, evidence=evidence)
    for item in lock["schedule"]:
        row = by_id[item["episode_id"]]
        await execute_pair(
            row=row,
            policies=policies,
            order=item["policy_order"],
            runner=verified_runner,
            maximum_costs={
                key: runtime.maximum_cost(row, policy) for key, policy in policies.items()
            },
            budget=ProductStudyBudget(directory / "budget"),
            stage="test",
            output=directory / "test-outcomes.jsonl",
            lock_hash=lock["lock_hash"],
            lock=lock,
            allow_bounded_unknown=True,
        )
        print(json.dumps({"stage": "test", "completed_pair": row.episode_id}), flush=True)
    render_report(directory)
    if stop_after == "test":
        return
    _, _, smoke_plans = validate_contract(directory, ["smoke"])
    runtime.baseline_plans.update(smoke_plans)
    for row in load_rows(directory, ["smoke"]):
        await collect_row(
            row, policies, stage="smoke", runtime=runtime, directory=directory, contract=contract, allow_bounded_unknown=True
        )
    render_report(directory)


def render_report(directory):
    budget = ProductStudyBudget(directory / "budget").summary()
    stages = {
        stage: outcomes(directory / f"{stage}-outcomes.jsonl")
        for stage in ("pilot", "development", "test", "smoke")
    }
    # Export every incurred attempt, including known charges inside an episode
    # whose total remains unknown. Outcome-only sums would hide those charges.
    raw_attempts = read_events(directory / "attempts.jsonl")
    attempts = {}
    for event in raw_attempts:
        if event.get("event") == "attempt_started":
            if event["attempt_id"] in attempts:
                raise ValueError("duplicate attempt start")
            attempts[event["attempt_id"]] = dict(event)
        elif event.get("event") == "attempt_finished":
            if event["attempt_id"] not in attempts or attempts[event["attempt_id"]].get("event") == "attempt_finished":
                raise ValueError("unmatched or duplicate attempt terminal")
            attempts[event["attempt_id"]].update(event)
    for values in stages.values():
        for value in values:
            reconciliation = value.evidence.get("cost_reconciliation", {})
            for attempt_id in reconciliation.get("attempt_ids", []):
                if attempt_id not in attempts or attempts[attempt_id].get("cost_usd") is not None:
                    raise ValueError("attempt reconciliation is inconsistent")
                attempts[attempt_id].update({"cost_usd": 0.0, "cost_known": True, "cost_basis": reconciliation["basis"], "original_cost_usd": None})
    ledger = {"attempts": list(attempts.values()), "known_incurred_cost_usd": sum(a["cost_usd"] for a in attempts.values() if a.get("cost_usd") is not None),
              "unknown_charge_attempts": sum(a.get("cost_usd") is None for a in attempts.values()),
              "provider_billed_cost_usd": sum(a.get("billed_cost_usd") or 0 for a in attempts.values()),
              "zero_price_bounded_failures": sum(a.get("cost_basis") == "frozen_zero_charge_upper_bound" for a in attempts.values()),
              "budget": budget}
    atomic_json(directory / "spend-ledger.json", ledger)
    results = {
        stage: {
            "episodes": len(values),
            "known_cost_usd": sum(x.cost_usd for x in values if x.cost_usd is not None),
            "unknown_cost_episodes": sum(x.cost_usd is None for x in values),
            "successes": sum(x.success is True for x in values),
            "ungradable": sum(x.success is None for x in values),
            "operational_failures": sum(x.status != "completed" for x in values),
        }
        for stage, values in stages.items()
    }
    import statistics
    family_metrics = {}
    for stage, values in stages.items():
        family_metrics[stage] = {}
        for family in FAMILIES:
            policies = sorted({value.policy_id for value in values if value.family == family})
            family_metrics[stage][family] = {}
            for policy in policies:
                sample = [value for value in values if value.family == family and value.policy_id == policy]
                latencies = sorted(value.latency_ms for value in sample)
                family_metrics[stage][family][policy] = {"episodes": len(sample), "independent_groups": len({v.group_id for v in sample}),
                    "success_rate": statistics.mean(v.success for v in sample) if all(v.success is not None for v in sample) else None,
                    "ungradable": sum(v.success is None for v in sample),
                    "mean_cost_usd": statistics.mean(v.cost_usd for v in sample) if all(v.cost_usd is not None for v in sample) else None,
                    "unknown_cost_episodes": sum(v.cost_usd is None for v in sample),
                    "p50_latency_ms": statistics.median(latencies), "p95_latency_ms": latencies[max(0, math.ceil(.95 * len(latencies)) - 1)],
                    "operational_failures": sum(v.status != "completed" for v in sample), "model_attempts": sum(v.attempts for v in sample)}
    inference = {"status": "inconclusive", "reason": "locked paired test not completed"}
    protocol_verified = False
    protocol_reason = "The final test has not been locked."
    independent_test_groups = None
    power = None
    if (directory / "test-lock.json").exists():
        lock = read_json(directory / "test-lock.json")
        wanted = {item["episode_id"] for item in lock["schedule"]}
        rows = [row for row in load_rows(directory, ["test"]) if row.episode_id in wanted]
        inference = paired_report(rows, stages["test"], lock=lock)
        independent_test_groups = len({row.group_id for row in rows})
        power = dict(lock.get("power", {}))
        if independent_test_groups and power.get("used_variance") is not None:
            z = statistics.NormalDist().inv_cdf(.95) + statistics.NormalDist().inv_cdf(.8)
            power.update({"frozen_independent_groups": independent_test_groups,
                "available_sample_supports_target": independent_test_groups >= power["required_independent_groups"],
                "approximate_detectable_effect_at_frozen_n": z * math.sqrt(power["used_variance"] / independent_test_groups)})
        if not (directory / "power-plan.json").exists():
            write_once(directory / "power-plan.json", {**power, "derived_from_lock_hash": lock["lock_hash"], "report_derivation_does_not_change_locked_plan": True})
        protocol_reason = "This implementation run did not freeze a complete source snapshot before its final test."
        if lock.get("protocol"):
            from theriad.routing.episode_protocol import verify_protocol
            try:
                verify_protocol(lock["protocol"], check_live=False)
                expected = lock["protocol"]["manifest_hash"]
                protocol_verified = bool(stages["test"]) and all(value.evidence.get("protocol_manifest_hash") == expected and not value.evidence.get("protocol_verification_failed") for value in stages["test"])
                protocol_reason = "Every recorded test outcome matches the frozen source manifest." if protocol_verified else "Test outcomes lack matching source manifest evidence."
            except (ValueError, OSError) as error:
                protocol_reason = f"Frozen source verification failed: {type(error).__name__}."
    decision = "improved" if inference.get("status") == "improved" and protocol_verified else "inconclusive"
    primary = None
    if inference.get("success_difference") is not None:
        measured = inference["families"]
        primary = {"baseline_success": statistics.mean(measured[f]["baseline"]["success"] for f in FAMILIES),
            "challenger_success": statistics.mean(measured[f]["challenger"]["success"] for f in FAMILIES),
            "baseline_cost_usd": statistics.mean(measured[f]["baseline"]["mean_cost_usd"] for f in FAMILIES),
            "challenger_cost_usd": statistics.mean(measured[f]["challenger"]["mean_cost_usd"] for f in FAMILIES)}
    report = {
        "version": VERSION,
        "decision": decision,
        "inference": inference,
        "primary_metrics": primary,
        "spend": budget,
        "all_known_incurred_cost_usd": ledger["known_incurred_cost_usd"],
        "unknown_charge_attempts": ledger["unknown_charge_attempts"],
        "stages": results,
        "observed_family_metrics": family_metrics,
        "protocol_verified": protocol_verified,
        "protocol_reason": protocol_reason,
        "power": power,
        "production_changed": False,
        "public_request_format_changed": False,
        "coverage": {
            "equal_weight_families": list(FAMILIES),
            "live_user_route_verified": False,
            "real_browser_episode_verified": False,
        },
        "limitations": [
            f"The locked test has {independent_test_groups} independent groups; consult the power plan before interpreting a five-point target." if independent_test_groups is not None else "No final-test sample has been locked yet.",
            protocol_reason,
            "Product fixture controls omit account HTTP gates, personal memory and live circuit state; no real-traffic cost claim.",
            "Terminal project agents and frozen search evidence do not establish live browser or live web-search performance.",
            "Deterministic structured grading covers frozen criteria, not broad open-ended conversational quality.",
            "No production promotion occurs; a favorable point estimate is not an improvement claim.",
        ],
        "reproduction": f".venv/bin/python scripts/run_router_episode_study.py report --directory {directory}",
    }
    if (directory / "blocked.json").exists():
        report["blocked"] = read_json(directory / "blocked.json")
    atomic_json(directory / "report.json", report)
    atomic_json(directory / "route-smoke-evidence.json", {"status": "not_run" if not stages["smoke"] else "recorded",
        "live_user_http_route_verified": False, "browser_verified": False,
        "execution_layer": "real_product_core_with_resettable_fixtures",
        "outcomes": [asdict(value) for value in stages["smoke"]]})
    lines = [
        "# Router episode study",
        "",
        f"Result: **{decision}**. Production routing remains unchanged.",
        "",
        f"Committed study spend (including unresolved reservations): **${budget['committed_usd']:.6f} / $100**.",
        f"Known incurred attempt charges: **${ledger['known_incurred_cost_usd']:.8f}**; unresolved attempt charges: **{ledger['unknown_charge_attempts']}**.",
        "",
        "| Stage | Episodes | Successes | Known episode spend | Unknown charges | Operational failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, value in results.items():
        lines.append(
            f"| {stage} | {value['episodes']} | {value['successes']} | ${value['known_cost_usd']:.6f} | {value['unknown_cost_episodes']} | {value['operational_failures']} |"
        )
    if primary is not None:
        lines.extend(["", "## Locked test observations", "",
            f"Baseline success: **{primary['baseline_success']:.0%}**; challenger success: **{primary['challenger_success']:.0%}**.",
            f"Mean episode cost: **${primary['baseline_cost_usd']:.8f}** baseline and **${primary['challenger_cost_usd']:.8f}** challenger.", "",
            "| Family | Baseline success | Challenger success | Baseline mean cost | Challenger mean cost | Baseline p50 / p95 seconds | Challenger p50 / p95 seconds |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for family, metrics in inference["families"].items():
            a, b = metrics["baseline"], metrics["challenger"]
            lines.append(f"| {family} | {a['success']:.0%} | {b['success']:.0%} | ${a['mean_cost_usd']:.8f} | ${b['mean_cost_usd']:.8f} | {a['latency_p50_ms']/1000:.1f} / {a['latency_p95_ms']/1000:.1f} | {b['latency_p50_ms']/1000:.1f} / {b['latency_p95_ms']/1000:.1f} |")
        lines.extend(["", f"Diagnostic one-sided 95% bounds: quality difference lower bound **{inference['success_lower_one_sided_95'] * 100:.1f} percentage points**; cost difference upper bound **${inference['cost_upper_one_sided_95_usd']:.8f} per episode**.",
            "These small-sample bootstrap bounds cannot establish improvement." if inference["bootstrap"]["bounds_are_diagnostic_only"] else "Both uncertainty bounds, complete accounting and a verified protocol are required for an improvement claim."])
    if power:
        lines.extend(["", f"The pilot power estimate called for **{power['required_independent_groups']:,} independent groups** for 80% power at a five-point lift; the frozen test had **{independent_test_groups}**."])
    lines.extend(
        [
            "",
            "All stage spending is reserved before dispatch. Unknown charges retain the full reservation and cannot support a cost claim. Interrupted trajectories stop collection.",
            "",
            "## Interpretation",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "## Reproduce",
            "",
            f"`{report['reproduction']}`",
            "",
            "Detailed paired intervals, family latency, policy identities, frozen fixtures and durable attempts are in adjacent JSON/JSONL files.",
        ]
    )
    if report.get("blocked"):
        lines.extend(["", f"Collection stopped: {report['blocked']['reason']}."])
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "reconcile-free"))
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "benchmark_results" / "product_episode_study_v1"
    )
    parser.add_argument("--alternatives", nargs="+", default=list(DEFAULT_ALTERNATIVES))
    parser.add_argument("--corpus", type=Path, help="Optional reviewed Episode JSONL with grouped splits; default is the underpowered seed")
    parser.add_argument(
        "--stop-after",
        choices=("pilot", "development", "selection", "test", "smoke"),
        default="smoke",
    )
    args = parser.parse_args()
    if len(args.alternatives) > 2 or len(set(args.alternatives)) != len(args.alternatives):
        parser.error("at most two distinct alternatives are permitted")
    args.directory = args.directory.resolve()
    try:
        if args.command == "prepare":
            asyncio.run(prepare(args.directory, args.alternatives, args.corpus))
        elif args.command == "run":
            asyncio.run(run_study(args.directory, stop_after=args.stop_after))
        elif args.command == "reconcile-free":
            print(json.dumps({"reconciled_episodes": reconcile_free_failures(args.directory)}))
            render_report(args.directory)
        else:
            render_report(args.directory)
        blocked = args.directory / "blocked.json"
        if args.command == "run" and blocked.exists():
            append_event(args.directory / "execution-history.jsonl", {"event": "prior_stop_resolved", "stop": read_json(blocked)})
            blocked.unlink()
            render_report(args.directory)
    except Exception as error:
        # Provider errors may contain credentials or request bodies. Persist
        # only local study errors verbatim, never raw network exception text.
        safe = (
            str(error)
            if isinstance(error, (ValueError, StudyBudgetExceeded))
            else type(error).__name__
        )
        atomic_json(
            args.directory / "blocked.json",
            {
                "reason": safe,
                "error_type": type(error).__name__,
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if (args.directory / "contract.json").exists():
            render_report(args.directory)
        print(f"Study stopped: {safe}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
