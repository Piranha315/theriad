#!/usr/bin/env python3
"""Migrate a stopped Router V2 study to a monotonic reservation-only plan.

The migration is deliberately narrow: call identities and every non-reservation
call field must remain unchanged, reservation maxima may only increase for an
explicit model/task-format allowlist, and one terminal ambiguous reservation is
settled at the revised maximum without making that call retryable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from theriad.routing.execution_plan import canonical_hash, load_execution_plan  # noqa: E402


class EnvelopeMigrationError(RuntimeError):
    """Raised when a reservation-only migration cannot be proven safe."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_key(call: Any) -> tuple[str, str]:
    return call.prompt_id, call.model


def _call_without_maximum(call: Any) -> dict[str, Any]:
    value = call.to_dict()
    value.pop("maximum_cost_usd", None)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EnvelopeMigrationError(f"expected an object in {path}")
    return value


def build_migration(
    *,
    old_plan_path: Path,
    new_plan_path: Path,
    state_path: Path,
    ledger_path: Path,
    terminal_prompt_id: str,
    terminal_model: str,
    terminal_reservation_id: str,
    allowed_increases: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old_plan = load_execution_plan(old_plan_path)
    new_plan = load_execution_plan(new_plan_path)
    if old_plan.phase != new_plan.phase or old_plan.corpus_hash != new_plan.corpus_hash:
        raise EnvelopeMigrationError("phase or corpus changed")
    if old_plan.max_attempts != new_plan.max_attempts:
        raise EnvelopeMigrationError("max attempts changed")
    if old_plan.price_snapshot != new_plan.price_snapshot:
        raise EnvelopeMigrationError("price snapshot changed")
    if old_plan.executor_manifest_hash != new_plan.executor_manifest_hash:
        raise EnvelopeMigrationError("executor manifest changed")
    if old_plan.anchor_boundary_hash != new_plan.anchor_boundary_hash:
        raise EnvelopeMigrationError("anchor boundary changed")

    old_calls = {_call_key(call): call for call in old_plan.calls}
    new_calls = {_call_key(call): call for call in new_plan.calls}
    if old_calls.keys() != new_calls.keys():
        raise EnvelopeMigrationError("call identities changed")

    changed: list[dict[str, Any]] = []
    for key, old_call in old_calls.items():
        new_call = new_calls[key]
        if _call_without_maximum(old_call) != _call_without_maximum(new_call):
            raise EnvelopeMigrationError(f"non-reservation call fields changed for {key}")
        old_maximum = float(old_call.maximum_cost_usd)
        new_maximum = float(new_call.maximum_cost_usd)
        if new_maximum + 1e-12 < old_maximum:
            raise EnvelopeMigrationError(f"reservation decreased for {key}")
        if abs(new_maximum - old_maximum) <= 1e-12:
            continue
        target = f"{new_call.model}@{new_call.task_format}"
        if target not in allowed_increases:
            raise EnvelopeMigrationError(f"unapproved reservation increase for {target}")
        changed.append(
            {
                "prompt_id": new_call.prompt_id,
                "model": new_call.model,
                "task_format": new_call.task_format,
                "old_maximum_cost_usd": old_maximum,
                "new_maximum_cost_usd": new_maximum,
            }
        )
    if not changed:
        raise EnvelopeMigrationError("new plan contains no reservation increase")

    terminal_key = (terminal_prompt_id, terminal_model)
    if terminal_key not in new_calls:
        raise EnvelopeMigrationError("terminal call is absent from the new plan")
    terminal_new_maximum = float(new_calls[terminal_key].maximum_cost_usd)

    state = _load_json(state_path)
    if state.get("plan_hash") != old_plan.execution_plan_hash:
        raise EnvelopeMigrationError("state does not belong to the old executable plan")
    if state.get("corpus_hash") != old_plan.corpus_hash:
        raise EnvelopeMigrationError("state corpus changed")
    run = (
        ((state.get("rows") or {}).get(terminal_prompt_id) or {})
        .get("runs", {})
        .get(terminal_model)
    )
    if not isinstance(run, dict) or run.get("status") != "error":
        raise EnvelopeMigrationError("terminal run is not a persisted error")
    if run.get("reservation_id") != terminal_reservation_id:
        raise EnvelopeMigrationError("terminal run reservation does not match")
    if int(run.get("api_call_count") or 0) < 1:
        raise EnvelopeMigrationError("terminal run has no provider invocation")

    ledger = _load_json(ledger_path)
    entries = ledger.get("entries")
    if ledger.get("version") != 1 or not isinstance(entries, list):
        raise EnvelopeMigrationError("budget ledger schema is invalid")
    unsettled = [entry for entry in entries if entry.get("actual_cost_usd") is None]
    if len(unsettled) != 1 or unsettled[0].get("reservation_id") != terminal_reservation_id:
        raise EnvelopeMigrationError("terminal reservation is not the sole unsettled entry")
    reservation = unsettled[0]
    expected_key = f"{old_plan.phase}:{terminal_prompt_id}:{terminal_model}"
    if reservation.get("idempotency_key") != expected_key:
        raise EnvelopeMigrationError("terminal reservation idempotency key changed")
    old_reserved_maximum = float(reservation["maximum_cost_usd"])
    if terminal_new_maximum + 1e-12 < old_reserved_maximum:
        raise EnvelopeMigrationError("terminal reservation maximum decreased")

    proof_payload = {
        "schema_version": 1,
        "migration_kind": "monotonic_reservation_only",
        "from_execution_plan_hash": old_plan.execution_plan_hash,
        "to_execution_plan_hash": new_plan.execution_plan_hash,
        "corpus_hash": old_plan.corpus_hash,
        "call_count": len(old_calls),
        "call_keys_unchanged": True,
        "non_reservation_call_fields_unchanged": True,
        "changed_reservation_count": len(changed),
        "allowed_increases": sorted(allowed_increases),
        "changed_reservations_sha256": canonical_hash(changed),
        "terminal_reconciliation": {
            "prompt_id": terminal_prompt_id,
            "model": terminal_model,
            "reservation_id": terminal_reservation_id,
            "original_maximum_cost_usd": old_reserved_maximum,
            "conservative_hold_usd": terminal_new_maximum,
            "retry_authorized": False,
        },
    }
    proof_hash = canonical_hash(proof_payload)
    migrated_at = datetime.now(timezone.utc).isoformat()
    migration_record = {
        **proof_payload,
        "proof_hash": proof_hash,
        "migrated_at": migrated_at,
        "old_execution_plan_file_sha256": _file_sha256(old_plan_path),
        "new_execution_plan_file_sha256": _file_sha256(new_plan_path),
    }

    reservation["maximum_cost_usd"] = terminal_new_maximum
    reservation["actual_cost_usd"] = terminal_new_maximum
    reservation["reservation_envelope_migration_proof"] = proof_hash
    run["conservative_accounting_hold_usd"] = terminal_new_maximum
    run["reservation_reconciled_at"] = migrated_at
    run["reservation_envelope_migration_proof"] = proof_hash
    run["retry_authorized"] = False
    state["plan_hash"] = new_plan.execution_plan_hash
    state.setdefault("plan_migrations", []).append(migration_record)

    report = {
        **migration_record,
        "status": "ready_to_apply",
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "test_outcomes_opened": False,
        "router_v2_enabled": False,
    }
    return state, ledger, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-execution-plan", type=Path, required=True)
    parser.add_argument("--new-execution-plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--terminal-prompt-id", required=True)
    parser.add_argument("--terminal-model", required=True)
    parser.add_argument("--terminal-reservation-id", required=True)
    parser.add_argument("--allow-increase", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    state, ledger, report = build_migration(
        old_plan_path=args.old_execution_plan,
        new_plan_path=args.new_execution_plan,
        state_path=args.state,
        ledger_path=args.ledger,
        terminal_prompt_id=args.terminal_prompt_id,
        terminal_model=args.terminal_model,
        terminal_reservation_id=args.terminal_reservation_id,
        allowed_increases=frozenset(args.allow_increase),
    )
    if args.apply:
        if not args.yes:
            raise EnvelopeMigrationError("--apply requires --yes")
        _atomic_json(args.ledger, ledger)
        _atomic_json(args.state, state)
        report["status"] = "applied"
    _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
