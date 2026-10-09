"""Atomic hard-cap ledger for paid Router V2 data collection."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StudyBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class StudyBudgetConfig:
    hard_cap_usd: float = 25.0
    test_reserve_usd: float = 10.0

    def __post_init__(self) -> None:
        if self.hard_cap_usd <= 0:
            raise ValueError("hard cap must be positive")
        if not 0 <= self.test_reserve_usd < self.hard_cap_usd:
            raise ValueError("test reserve must be non-negative and below the hard cap")


class StudyBudgetLedger:
    """Reserve worst-case call cost before execution, then settle actual cost."""

    def __init__(self, path: str | Path, config: StudyBudgetConfig | None = None):
        self.path = Path(path)
        self.config = config or StudyBudgetConfig()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "entries": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudyBudgetExceeded("study budget ledger is unreadable") from exc
        if raw.get("version") != 1 or not isinstance(raw.get("entries"), list):
            raise StudyBudgetExceeded("study budget ledger has an invalid schema")
        return raw

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write(self, raw: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(raw, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _committed(entry: dict[str, Any]) -> float:
        actual = entry.get("actual_cost_usd")
        return float(actual if actual is not None else entry["maximum_cost_usd"])

    def reservation(self, reservation_id: str) -> dict[str, Any] | None:
        """Return one reservation without exposing any credential material."""

        with self._locked():
            raw = self._load()
            entry = next(
                (row for row in raw["entries"] if row.get("reservation_id") == reservation_id),
                None,
            )
            return dict(entry) if entry is not None else None

    def status(self) -> dict[str, float]:
        with self._locked():
            raw = self._load()
            if not self.path.exists():
                self._write(raw)
            total = sum(self._committed(entry) for entry in raw["entries"])
            non_test = sum(
                self._committed(entry)
                for entry in raw["entries"]
                if entry["phase"] != "test"
            )
        return {
            "committed_usd": total,
            "non_test_committed_usd": non_test,
            "remaining_usd": max(0.0, self.config.hard_cap_usd - total),
            "non_test_remaining_usd": max(
                0.0,
                self.config.hard_cap_usd - self.config.test_reserve_usd - non_test,
            ),
        }

    def accounting(self) -> dict[str, float]:
        """Expose reservation and realized totals as separate quantities."""

        with self._locked():
            raw = self._load()
            if not self.path.exists():
                self._write(raw)
            reserved = sum(float(entry["maximum_cost_usd"]) for entry in raw["entries"])
            outstanding = sum(
                float(entry["maximum_cost_usd"])
                for entry in raw["entries"]
                if entry.get("actual_cost_usd") is None
            )
            realized = sum(
                float(entry.get("actual_cost_usd") or 0.0)
                for entry in raw["entries"]
                if entry.get("actual_cost_usd") is not None
            )
            entries = len(raw["entries"])
        return {
            "reservation_count": float(entries),
            "reserved_cost_usd": reserved,
            "outstanding_reserved_cost_usd": outstanding,
            "realized_provider_cost_usd": realized,
        }

    def reserve(
        self,
        *,
        phase: str,
        model: str,
        maximum_cost_usd: float,
        idempotency_key: str | None = None,
        potential_api_calls: int = 1,
        expected_turns: int = 1,
        max_turns: int = 1,
        expected_trajectory_tokens: int = 0,
        max_trajectory_tokens: int = 0,
    ) -> str:
        if phase not in {"train", "validation", "test"}:
            raise ValueError(f"invalid study phase {phase!r}")
        maximum = float(maximum_cost_usd)
        if maximum < 0:
            raise ValueError("maximum call cost cannot be negative")
        if min(potential_api_calls, expected_turns, max_turns) < 1:
            raise ValueError("trajectory call and turn counts must be positive")
        if min(expected_trajectory_tokens, max_trajectory_tokens) < 0:
            raise ValueError("trajectory token counts cannot be negative")
        with self._locked():
            raw = self._load()
            if idempotency_key:
                existing = next(
                    (
                        row
                        for row in raw["entries"]
                        if row.get("idempotency_key") == idempotency_key
                    ),
                    None,
                )
                if existing is not None:
                    if (
                        existing.get("phase") != phase
                        or existing.get("model") != model
                        or abs(float(existing["maximum_cost_usd"]) - maximum) > 1e-12
                        or int(existing.get("potential_api_calls", 1))
                        != int(potential_api_calls)
                        or int(existing.get("expected_turns", 1))
                        != int(expected_turns)
                        or int(existing.get("max_turns", 1)) != int(max_turns)
                        or int(existing.get("expected_trajectory_tokens", 0))
                        != int(expected_trajectory_tokens)
                        or int(existing.get("max_trajectory_tokens", 0))
                        != int(max_trajectory_tokens)
                    ):
                        raise StudyBudgetExceeded(
                            f"idempotency key {idempotency_key!r} was reused with a different reservation or trajectory contract"
                        )
                    return str(existing["reservation_id"])
            total = sum(self._committed(entry) for entry in raw["entries"])
            non_test = sum(
                self._committed(entry)
                for entry in raw["entries"]
                if entry["phase"] != "test"
            )
            if total + maximum > self.config.hard_cap_usd + 1e-12:
                raise StudyBudgetExceeded("$25 study hard cap would be exceeded")
            if (
                phase != "test"
                and non_test + maximum
                > self.config.hard_cap_usd - self.config.test_reserve_usd + 1e-12
            ):
                raise StudyBudgetExceeded("the untouched-test reserve would be consumed")
            reservation_id = str(uuid.uuid4())
            raw["entries"].append(
                {
                    "reservation_id": reservation_id,
                    "phase": phase,
                    "model": model,
                    "maximum_cost_usd": maximum,
                    "actual_cost_usd": None,
                    "idempotency_key": idempotency_key,
                    "potential_api_calls": int(potential_api_calls),
                    "expected_turns": int(expected_turns),
                    "max_turns": int(max_turns),
                    "expected_trajectory_tokens": int(expected_trajectory_tokens),
                    "max_trajectory_tokens": int(max_trajectory_tokens),
                }
            )
            self._write(raw)
        return reservation_id

    def settle(self, reservation_id: str, actual_cost_usd: float) -> None:
        actual = float(actual_cost_usd)
        if actual < 0:
            raise ValueError("actual cost cannot be negative")
        with self._locked():
            raw = self._load()
            entry = next(
                (row for row in raw["entries"] if row["reservation_id"] == reservation_id),
                None,
            )
            if entry is None:
                raise KeyError(reservation_id)
            if actual > float(entry["maximum_cost_usd"]) + 1e-12:
                raise StudyBudgetExceeded("provider cost exceeded the pre-authorized maximum")
            entry["actual_cost_usd"] = actual
            self._write(raw)
