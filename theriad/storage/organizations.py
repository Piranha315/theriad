"""Personal-organization ownership for authenticated Theriad accounts.

This is deliberately small foundation work. Resources remain keyed by the
existing local ``user_id`` columns for now; the membership row gives every
registered account an explicit owner boundary so a later team-organization
migration has a stable identity to build on. It never claims or rewrites
legacy rows whose ``user_id`` is NULL.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from theriad.storage.database import get_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_name(label: str | None) -> str:
    clean = " ".join((label or "").split()).strip()
    if not clean:
        return "Theriad workspace"
    return f"{clean}'s workspace"


def get_personal_membership(user_id: str) -> dict[str, Any] | None:
    """Return the user's personal organization membership, if it exists."""
    db = get_db()
    row = db.execute(
        """SELECT m.organization_id, m.role, o.name, o.kind
           FROM organization_memberships m
           JOIN organizations o ON o.id = m.organization_id
           WHERE m.user_id = ? AND m.personal = 1 AND o.kind = 'personal'
           ORDER BY m.created_at ASC
           LIMIT 1""",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def ensure_personal_organization(user_id: str, label: str | None = None) -> dict[str, Any]:
    """Create or return the one personal organization owned by ``user_id``.

    Hosted mode persists the organization and membership in Supabase; local
    mode retains the SQLite adapter for development and offline use.
    """
    existing = get_personal_membership(user_id)
    try:
        from theriad.storage.hosted import HostedStorageUnavailable, hosted_data_plane_enabled, hosted_repositories

        if hosted_data_plane_enabled():
            repos = hosted_repositories()
            organization_id = str(existing["organization_id"]) if existing else None
            if not organization_id:
                prior_memberships = repos["memberships"].client.list_rows(
                    "organization_memberships",
                    filters={"user_id": f"eq.{user_id}", "personal": "eq.true"},
                    limit=1,
                )
                organization_id = str(prior_memberships[0]["organization_id"]) if prior_memberships else str(uuid.uuid4())
            organization_name = str(existing.get("name") or _default_name(label)) if existing else _default_name(label)
            organizations = repos["organizations"].client.list_rows(
                "organizations", filters={"id": f"eq.{organization_id}"}, limit=1
            )
            if organizations:
                organization_name = str(organizations[0].get("name") or organization_name)
            else:
                repos["organizations"].client.insert_row(
                    "organizations",
                    {"id": organization_id, "name": organization_name, "kind": "personal"},
                )
            memberships = repos["memberships"].client.list_rows(
                "organization_memberships",
                filters={
                    "organization_id": f"eq.{organization_id}",
                    "user_id": f"eq.{user_id}",
                },
                limit=1,
            )
            if memberships:
                role = memberships[0].get("role") or "owner"
                if role != "owner":
                    repos["memberships"].client.update_rows(
                        "organization_memberships",
                        filters={
                            "organization_id": f"eq.{organization_id}",
                            "user_id": f"eq.{user_id}",
                        },
                        values={"role": "owner", "personal": True},
                    )
                    role = "owner"
            else:
                repos["memberships"].client.insert_row(
                    "organization_memberships",
                    {
                        "organization_id": organization_id,
                        "user_id": user_id,
                        "role": "owner",
                        "personal": True,
                    },
                )
                role = "owner"
            return {
                "organization_id": organization_id,
                "role": role,
                "name": organization_name,
                "kind": "personal",
            }
    except HostedStorageUnavailable:
        # Hosted mode must not downgrade to a local organization when the
        # configured Supabase data plane is unavailable.
        raise

    if existing:
        return existing

    db = get_db()
    organization_id = str(uuid.uuid4())
    now = _now()
    try:
        db.execute(
            "INSERT INTO organizations (id, name, kind, created_at) VALUES (?, ?, 'personal', ?)",
            (organization_id, _default_name(label), now),
        )
        db.execute(
            """INSERT INTO organization_memberships
               (organization_id, user_id, role, personal, created_at)
               VALUES (?, ?, 'owner', 1, ?)""",
            (organization_id, user_id, now),
        )
        db.commit()
    except sqlite3.IntegrityError:
        # A second request can race the first account bootstrap. The unique
        # personal-membership index makes the first commit authoritative; the
        # losing transaction must roll back its orphan organization and then
        # re-read the winner.
        db.rollback()
        existing = get_personal_membership(user_id)
        if existing:
            return existing
        raise
    membership = get_personal_membership(user_id)
    if membership is None:  # pragma: no cover - defensive: the insert just committed
        raise RuntimeError("Personal organization membership was not persisted.")
    return membership


def ensure_guest_organization(user_id: str) -> dict[str, Any]:
    """Create or return a hosted staging organization for a guest session."""
    from theriad.storage.hosted import hosted_data_plane_enabled, hosted_repositories

    if not hosted_data_plane_enabled():
        return {"organization_id": None, "role": None, "name": "Guest workspace", "kind": "personal"}
    repos = hosted_repositories()
    memberships = repos["memberships"].client.list_rows(
        "organization_memberships",
        filters={"user_id": f"eq.{user_id}", "personal": "eq.true"},
        limit=1,
    )
    if memberships:
        organization_id = str(memberships[0]["organization_id"])
        role = memberships[0].get("role") or "member"
        organizations = repos["organizations"].client.list_rows(
            "organizations", filters={"id": f"eq.{organization_id}"}, limit=1
        )
        return {
            "organization_id": organization_id,
            "role": role,
            "name": organizations[0].get("name") if organizations else "Guest workspace",
            "kind": organizations[0].get("kind", "personal") if organizations else "personal",
        }
    organization_id = str(uuid.uuid4())
    repos["organizations"].client.insert_row(
        "organizations", {"id": organization_id, "name": "Guest workspace", "kind": "personal"}
    )
    repos["memberships"].client.insert_row(
        "organization_memberships",
        {
            "organization_id": organization_id,
            "user_id": user_id,
            "role": "member",
            "personal": True,
        },
    )
    return {
        "organization_id": organization_id,
        "role": "member",
        "name": "Guest workspace",
        "kind": "personal",
    }
