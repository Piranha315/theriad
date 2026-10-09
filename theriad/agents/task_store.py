"""Durable storage adapters for the canonical TaskEngine supervisor.

Local JSON remains the zero-configuration development backend. Supabase uses
server-only PostgREST calls and database RPCs; credentials are never returned
to clients or written to task snapshots.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


class TaskStoreError(RuntimeError):
    pass


class TaskStore(Protocol):
    name: str

    def load_snapshots(self) -> list[dict[str, Any]]: ...
    def load_snapshot(self, task_id: str) -> dict[str, Any] | None: ...
    def load_events(self, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]: ...
    def persist(
        self,
        snapshot: dict[str, Any],
        event: dict[str, Any] | None = None,
    ) -> None: ...
    def claim(self, task_id: str, supervisor_id: str, lease_seconds: int) -> bool: ...
    def heartbeat(self, task_id: str, supervisor_id: str, lease_seconds: int) -> bool: ...
    def release(self, task_id: str, supervisor_id: str) -> None: ...
    def status(self) -> dict[str, Any]: ...


class LocalTaskStore:
    """Atomic local snapshots plus append-only JSONL events."""

    name = "local"

    def __init__(self, event_dir: str | Path) -> None:
        self.event_dir = Path(event_dir)
        self._leases: dict[str, tuple[str, float]] = {}
        self._lock = threading.RLock()

    def _snapshot_path(self, task_id: str) -> Path:
        return self.event_dir / f"{task_id}.snapshot.json"

    def load_snapshots(self) -> list[dict[str, Any]]:
        if not self.event_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.event_dir.glob("*.snapshot.json")):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return rows

    def load_snapshot(self, task_id: str) -> dict[str, Any] | None:
        path = self._snapshot_path(task_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def load_events(self, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        rows: dict[str, list[dict[str, Any]]] = {}
        for task_id in task_ids:
            path = self.event_dir / f"{task_id}.jsonl"
            if not path.exists():
                continue
            events = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            rows[task_id] = events
        return rows

    def persist(
        self,
        snapshot: dict[str, Any],
        event: dict[str, Any] | None = None,
    ) -> None:
        task_id = str(snapshot["task"]["task_id"])
        with self._lock:
            self.event_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            target = self._snapshot_path(task_id)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(
                json.dumps(snapshot, separators=(",", ":")),
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            if event is not None:
                with (self.event_dir / f"{task_id}.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    def claim(self, task_id: str, supervisor_id: str, lease_seconds: int) -> bool:
        now = time.monotonic()
        with self._lock:
            owner, expires = self._leases.get(task_id, ("", 0.0))
            if owner and owner != supervisor_id and expires > now:
                return False
            self._leases[task_id] = (supervisor_id, now + lease_seconds)
            return True

    def heartbeat(self, task_id: str, supervisor_id: str, lease_seconds: int) -> bool:
        with self._lock:
            owner, _expires = self._leases.get(task_id, ("", 0.0))
            if owner != supervisor_id:
                return False
            self._leases[task_id] = (supervisor_id, time.monotonic() + lease_seconds)
            return True

    def release(self, task_id: str, supervisor_id: str) -> None:
        with self._lock:
            if self._leases.get(task_id, (None,))[0] == supervisor_id:
                self._leases.pop(task_id, None)

    def status(self) -> dict[str, Any]:
        return {"backend": self.name, "configured": True, "shared": False}


@dataclass(frozen=True)
class SupabaseTaskStoreConfig:
    url: str
    secret_key: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> SupabaseTaskStoreConfig:
        url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
        secret = (
            os.environ.get("SUPABASE_SECRET_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or ""
        ).strip()
        if not url or not secret:
            raise TaskStoreError(
                "Supabase task storage requires SUPABASE_URL and the server-only "
                "SUPABASE_SECRET_KEY (or legacy SUPABASE_SERVICE_ROLE_KEY)."
            )
        return cls(url=url, secret_key=secret)


class SupabaseTaskStore:
    """Shared Postgres task store accessed through Supabase PostgREST/RPC."""

    name = "supabase"

    def __init__(self, config: SupabaseTaskStoreConfig) -> None:
        self.config = config

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.config.secret_key,
            "Content-Type": "application/json",
        }
        # Legacy service_role keys are JWTs and can establish the Postgres
        # role through Authorization. New sb_secret_ keys are gateway secrets,
        # not user JWTs, and belong only in apikey.
        if self.config.secret_key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self.config.secret_key}"
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        payload: Any = None,
        prefer: str | None = None,
    ) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self.config.url}{path}",
                params=params,
                json=payload,
                headers=self._headers(prefer=prefer),
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise TaskStoreError(
                f"Supabase task storage request failed ({method} {path})."
            ) from exc

    def load_snapshots(self) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            "/rest/v1/agent_tasks",
            params={
                "select": "snapshot",
                "order": "updated_at.desc",
                "limit": "1000",
            },
        )
        try:
            return [row["snapshot"] for row in response.json() if row.get("snapshot")]
        except (TypeError, ValueError, KeyError) as exc:
            raise TaskStoreError("Supabase returned invalid task snapshots.") from exc

    def load_snapshot(self, task_id: str) -> dict[str, Any] | None:
        response = self._request(
            "GET",
            "/rest/v1/agent_tasks",
            params={"select": "snapshot", "id": f"eq.{task_id}", "limit": "1"},
        )
        try:
            rows = response.json()
            return rows[0]["snapshot"] if rows else None
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise TaskStoreError("Supabase returned an invalid task snapshot.") from exc

    def load_events(self, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        # Bound URL size and avoid one request per task.
        for offset in range(0, len(task_ids), 100):
            chunk = task_ids[offset : offset + 100]
            if not chunk:
                continue
            response = self._request(
                "GET",
                "/rest/v1/agent_task_events",
                params={
                    "select": "event",
                    "task_id": f"in.({','.join(chunk)})",
                    "order": "task_id.asc,sequence.asc",
                    "limit": "10000",
                },
            )
            try:
                events = [row["event"] for row in response.json() if row.get("event")]
            except (TypeError, ValueError, KeyError) as exc:
                raise TaskStoreError("Supabase returned invalid task events.") from exc
            for event in events:
                grouped.setdefault(str(event["task_id"]), []).append(event)
        return grouped

    def persist(
        self,
        snapshot: dict[str, Any],
        event: dict[str, Any] | None = None,
    ) -> None:
        self._request(
            "POST",
            "/rest/v1/rpc/persist_agent_task",
            payload={"p_snapshot": snapshot, "p_event": event},
            prefer="return=minimal",
        )

    def claim(self, task_id: str, supervisor_id: str, lease_seconds: int) -> bool:
        response = self._request(
            "POST",
            "/rest/v1/rpc/claim_agent_task",
            payload={
                "p_task_id": task_id,
                "p_supervisor_id": supervisor_id,
                "p_lease_seconds": lease_seconds,
            },
        )
        return bool(response.json())

    def heartbeat(self, task_id: str, supervisor_id: str, lease_seconds: int) -> bool:
        response = self._request(
            "POST",
            "/rest/v1/rpc/heartbeat_agent_task",
            payload={
                "p_task_id": task_id,
                "p_supervisor_id": supervisor_id,
                "p_lease_seconds": lease_seconds,
            },
        )
        return bool(response.json())

    def release(self, task_id: str, supervisor_id: str) -> None:
        self._request(
            "POST",
            "/rest/v1/rpc/release_agent_task",
            payload={"p_task_id": task_id, "p_supervisor_id": supervisor_id},
            prefer="return=minimal",
        )

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "configured": True,
            "shared": True,
            "project_url_configured": bool(self.config.url),
        }


def task_store_configuration() -> dict[str, Any]:
    mode = (os.environ.get("THERIAD_TASK_STORE") or "auto").strip().lower()
    has_url = bool((os.environ.get("SUPABASE_URL") or "").strip())
    has_secret = bool(
        (
            os.environ.get("SUPABASE_SECRET_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or ""
        ).strip()
    )
    selected = "supabase" if mode == "supabase" or (mode == "auto" and has_url and has_secret) else "local"
    return {
        "mode": mode,
        "selected": selected,
        "supabase_url_configured": has_url,
        "supabase_secret_configured": has_secret,
        "shared": selected == "supabase",
    }


def supervisor_lease_seconds() -> int:
    try:
        value = int(os.environ.get("THERIAD_SUPERVISOR_LEASE_SECONDS", "120"))
    except ValueError:
        value = 120
    return max(5, min(value, 300))


def build_task_store(event_dir: str | Path) -> TaskStore:
    configuration = task_store_configuration()
    mode = configuration["mode"]
    if mode not in {"auto", "local", "supabase"}:
        raise TaskStoreError("THERIAD_TASK_STORE must be auto, local, or supabase.")
    if configuration["selected"] == "supabase":
        return SupabaseTaskStore(SupabaseTaskStoreConfig.from_environment())
    if mode == "supabase":
        # The explicit shared mode must never silently create split-brain local state.
        raise TaskStoreError("Supabase task storage is selected but not configured.")
    return LocalTaskStore(event_dir)
