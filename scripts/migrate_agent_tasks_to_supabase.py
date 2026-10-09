#!/usr/bin/env python3
"""Copy local TaskEngine snapshots/events to Supabase without deleting source files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from theriad.agents.task_store import (  # noqa: E402
    LocalTaskStore,
    SupabaseTaskStore,
    SupabaseTaskStoreConfig,
)


def migration_plan(source: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    local = LocalTaskStore(source)
    snapshots = [
        row
        for row in local.load_snapshots()
        if row.get("version") == 1 and row.get("task", {}).get("task_id")
    ]
    task_ids = [str(row["task"]["task_id"]) for row in snapshots]
    return snapshots, local.load_events(task_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/agent-tasks"),
        help="Local TaskEngine checkpoint directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write to Supabase. Without this flag, only print the migration plan.",
    )
    args = parser.parse_args()
    load_dotenv(_ROOT / ".env")

    snapshots, events = migration_plan(args.source)
    event_count = sum(len(rows) for rows in events.values())
    print(f"tasks={len(snapshots)} events={event_count} source={args.source}")
    if not args.apply:
        print("dry_run=true; rerun with --apply after applying the Supabase migration")
        return 0

    shared = SupabaseTaskStore(SupabaseTaskStoreConfig.from_environment())
    for snapshot in snapshots:
        task_id = str(snapshot["task"]["task_id"])
        shared.persist(snapshot)
        for event in events.get(task_id, []):
            shared.persist(snapshot, event)
    print("applied=true; local source files were preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
