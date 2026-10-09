"""Private local storage with transactional mutations. Cloud implementation is optional."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class Store(Protocol):
    def get(self, key: str) -> Any: ...
    def put(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    def mutate(self, key: str, fn: Callable[[Any], Any]) -> Any: ...


class LocalStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires REAL)"
            )
        os.chmod(self.path, 0o600)
        self.cleanup()

    def connection(self):
        return sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")

    def cleanup(self):
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM records WHERE expires IS NOT NULL AND expires<=?", (time.time(),)
            )

    def get(self, key):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value, expires FROM records WHERE key=?", (key,)
            ).fetchone()
        if not row or (row[1] is not None and row[1] <= time.time()):
            if row:
                self.cleanup()
            return None
        return json.loads(row[0])

    def put(self, key, value, ttl=None):
        with self.connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?)",
                (key, json.dumps(value, allow_nan=False), time.time() + ttl if ttl else None),
            )

    def mutate(self, key, fn):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value, expires FROM records WHERE key=?", (key,)
            ).fetchone()
            current = (
                json.loads(row[0]) if row and (row[1] is None or row[1] > time.time()) else None
            )
            result = fn(current)
            connection.execute(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?)",
                (
                    key,
                    json.dumps(result, allow_nan=False),
                    row[1] if row and current is not None else None,
                ),
            )
            return result
