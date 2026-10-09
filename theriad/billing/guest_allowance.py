"""Platform-paid allowance for guest chat (including web search).

Guests have no OpenRouter key, so their requests run on a platform key
(``THERIAD_GUEST_OPENROUTER_API_KEY``) until they have spent a small allowance.
After that, guest requests are refused with a message asking them to create an
account and connect their own key.

Guest accounts are free to create, so a per-guest cap alone is bypassed by
opening a private window. Spend is therefore capped three ways:

- per guest account, lifetime  (``THERIAD_GUEST_ALLOWANCE_USD``, default $1)
- per client IP, rolling 24h   (``THERIAD_GUEST_ALLOWANCE_PER_IP_DAILY_USD``, default $1)
- all guests, rolling 24h      (``THERIAD_GUEST_ALLOWANCE_GLOBAL_DAILY_USD``, default $10)

Client IPs are stored only as salted hashes.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Charged when the provider did not report a cost, so unknown spend still counts.
UNKNOWN_COST_ESTIMATE_USD = 0.01

EXHAUSTED_MESSAGE = (
    "You've used the free guest allowance. Create a free account and connect "
    "your OpenRouter key to keep chatting and searching the web."
)
UNAVAILABLE_MESSAGE = (
    "Free guest access is busy right now. Create a free account and connect "
    "your OpenRouter key to keep going."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS guest_platform_spend (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    client_hash TEXT NOT NULL,
    cost_usd    REAL NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_guest_spend_user ON guest_platform_spend(user_id);
CREATE INDEX IF NOT EXISTS idx_guest_spend_client ON guest_platform_spend(client_hash, created_at);
CREATE INDEX IF NOT EXISTS idx_guest_spend_created ON guest_platform_spend(created_at);
"""


def _usd(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, "") or default))
    except ValueError:
        return default


def platform_key() -> str | None:
    return os.environ.get("THERIAD_GUEST_OPENROUTER_API_KEY", "").strip() or None


def client_ip(request: Any) -> str:
    """The caller's IP; proxy headers are trusted only from a loopback peer."""
    peer = getattr(getattr(request, "client", None), "host", "") or ""
    try:
        from_proxy = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        from_proxy = False
    if from_proxy:
        headers = request.headers
        real_ip = (headers.get("x-real-ip") or "").strip()
        if real_ip:
            return real_ip
        forwarded = [part.strip() for part in (headers.get("x-forwarded-for") or "").split(",") if part.strip()]
        if forwarded:
            # nginx appends the address it saw; earlier entries are client-supplied.
            return forwarded[-1]
    return peer or "unknown"


def _client_hash(ip: str) -> str:
    salt = os.environ.get("THERIAD_SECRET_ENCRYPTION_KEY", "") or "theriad-guest-allowance"
    return hashlib.sha256(f"{salt}|{ip}".encode()).hexdigest()[:32]


def _db():
    from theriad.storage.database import get_db

    db = get_db()
    # Plain ``execute`` of idempotent DDL joins any open transaction instead of
    # committing it (``executescript`` would commit unrelated pending writes).
    for statement in filter(None, (part.strip() for part in _SCHEMA.split(";"))):
        db.execute(statement)
    return db


@dataclass(frozen=True)
class GuestAllowance:
    enabled: bool
    allowed: bool
    user_id: str | None = None
    client_hash: str = ""
    api_key: str | None = None
    spent_usd: float = 0.0
    limit_usd: float = 0.0
    reason: str = ""

    @property
    def message(self) -> str:
        return UNAVAILABLE_MESSAGE if self.reason == "global" else EXHAUSTED_MESSAGE

    @classmethod
    def for_request(cls, request: Any, *, user_id: str | None) -> "GuestAllowance":
        key = platform_key()
        if not key or not user_id:
            return cls(enabled=False, allowed=False)
        client_hash = _client_hash(client_ip(request))
        per_user = _usd("THERIAD_GUEST_ALLOWANCE_USD", 1.0)
        per_ip = _usd("THERIAD_GUEST_ALLOWANCE_PER_IP_DAILY_USD", per_user)
        global_daily = _usd("THERIAD_GUEST_ALLOWANCE_GLOBAL_DAILY_USD", 10.0)
        since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        db = _db()

        def total(where: str, params: tuple) -> float:
            row = db.execute(
                f"SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM guest_platform_spend WHERE {where}",
                params,
            ).fetchone()
            return float(row["spent"] if hasattr(row, "keys") else row[0])

        user_spent = total("user_id = ?", (user_id,))
        checks = (
            ("user", user_spent, per_user),
            ("ip", total("client_hash = ? AND created_at >= ?", (client_hash, since)), per_ip),
            ("global", total("created_at >= ?", (since,)), global_daily),
        )
        for reason, spent, limit in checks:
            if spent >= limit:
                return cls(True, False, user_id, client_hash, None, user_spent, per_user, reason)
        return cls(True, True, user_id, client_hash, key, user_spent, per_user)

    def record(self, cost_usd: float | None) -> None:
        if not self.enabled or not self.user_id:
            return
        cost = UNKNOWN_COST_ESTIMATE_USD if cost_usd is None else max(0.0, float(cost_usd))
        if cost <= 0:
            return
        db = _db()
        db.execute(
            "INSERT INTO guest_platform_spend (id, user_id, client_hash, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), self.user_id, self.client_hash, cost,
             datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
