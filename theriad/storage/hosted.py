"""Supabase-backed hosted data-plane boundary.

The application has a deliberately small SQLite adapter for local/dev mode.
Hosted mode uses this module as the explicit boundary for Postgres and Storage:
all writes carry both an organization id and a user id, and the server uses a
server-only Supabase key. The database migration still protects direct client
access with RLS; this key must never be placed in a Vite variable or returned
from a status endpoint.

The repository objects are intentionally boring wrappers around PostgREST.
Keeping the transport here means the API routes do not grow one-off Supabase
HTTP calls with subtly different ownership filters.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import quote
from typing import Any

import httpx


class HostedStorageUnavailable(RuntimeError):
    """Raised when the configured hosted data plane cannot be used safely."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def hosted_data_plane_enabled() -> bool:
    """Whether the process has opted into the Supabase hosted adapter.

    Keeping the switch explicit preserves local/dev SQLite compatibility. A
    hosted deployment should set both ``THERIAD_STORAGE_BACKEND=supabase`` and
    ``THERIAD_HOSTED_STORAGE_REQUIRED=true`` through its secret/config path.
    """
    backend = os.environ.get("THERIAD_STORAGE_BACKEND", "sqlite").strip().lower()
    if backend == "supabase" or _truthy(os.environ.get("THERIAD_HOSTED_STORAGE_REQUIRED")):
        return True
    # A hosted deployment is never allowed to silently fall back to SQLite.
    # Import lazily to keep this low-level adapter independent during startup.
    try:
        from theriad.settings.deployment import is_hosted_mode

        return is_hosted_mode()
    except Exception:
        return False


@dataclass(frozen=True)
class HostedStorageConfig:
    url: str | None
    service_role_key: str | None
    bucket: str
    backend: str
    required: bool

    @property
    def configured(self) -> bool:
        return bool(self.url and self.service_role_key)


def get_hosted_storage_config() -> HostedStorageConfig:
    return HostedStorageConfig(
        url=(os.environ.get("SUPABASE_URL") or "").strip().rstrip("/") or None,
        service_role_key=(
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_SERVER_KEY")
            or ""
        ).strip() or None,
        bucket=os.environ.get("THERIAD_SUPABASE_STORAGE_BUCKET", "theriad-files").strip()
        or "theriad-files",
        backend=os.environ.get("THERIAD_STORAGE_BACKEND", "sqlite").strip().lower() or "sqlite",
        required=hosted_data_plane_enabled(),
    )


def storage_status(*, probe: bool = False) -> dict[str, Any]:
    """Return safe configuration/health metadata; never return credentials."""
    config = get_hosted_storage_config()
    reachable: bool | None = None
    if probe and config.configured:
        try:
            SupabaseRestClient(config).probe()
            reachable = True
        except HostedStorageUnavailable:
            reachable = False
    return {
        "backend": config.backend,
        "required": config.required,
        "configured": config.configured,
        "reachable": reachable,
        "bucket": config.bucket,
    }


def assert_hosted_storage_ready(*, probe: bool = True) -> HostedStorageConfig:
    """Fail closed for a deployment that declares hosted storage required."""
    config = get_hosted_storage_config()
    if not config.required:
        raise HostedStorageUnavailable(
            "Hosted storage is not enabled; set THERIAD_STORAGE_BACKEND=supabase."
        )
    if not config.configured:
        raise HostedStorageUnavailable(
            "Supabase hosted storage is not configured on the server."
        )
    if probe:
        SupabaseRestClient(config).probe()
    return config


class SupabaseRestClient:
    """Minimal synchronous PostgREST/Storage client for FastAPI sync routes."""

    def __init__(self, config: HostedStorageConfig | None = None):
        self.config = config or get_hosted_storage_config()
        if not self.config.url or not self.config.service_role_key:
            raise HostedStorageUnavailable("Supabase hosted storage is not configured on the server.")

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.config.service_role_key or "",
            "Authorization": f"Bearer {self.config.service_role_key or ''}",
            "Content-Profile": "public",
            "Accept-Profile": "public",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        expected: tuple[int, ...] = (200, 201, 204),
    ) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self.config.url}{path}",
                params=params,
                json=payload,
                headers={**self._headers, **(headers or {})},
                timeout=12.0,
            )
        except httpx.HTTPError as exc:
            raise HostedStorageUnavailable("Supabase hosted storage is unavailable.") from exc
        if response.status_code not in expected:
            # Never include response text: Supabase errors can echo SQL,
            # object names, or operator configuration details.
            if response.status_code in {401, 403}:
                message = "Supabase hosted storage rejected the server credentials."
            else:
                message = "Supabase hosted storage returned an error."
            raise HostedStorageUnavailable(message)
        return response

    def probe(self) -> None:
        self._request(
            "GET",
            "/rest/v1/organizations",
            params={"select": "id", "limit": "1"},
            expected=(200,),
        )

    def list_rows(
        self,
        table: str,
        *,
        filters: dict[str, str] | None = None,
        select: str = "*",
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params = {"select": select, **(filters or {})}
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(max(1, min(int(limit), 1000)))
        response = self._request("GET", f"/rest/v1/{table}", params=params, expected=(200,))
        try:
            data = response.json()
        except ValueError as exc:
            raise HostedStorageUnavailable("Supabase returned invalid hosted data.") from exc
        return [dict(item) for item in data] if isinstance(data, list) else []

    def insert_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"/rest/v1/{table}",
            payload=row,
            headers={"Prefer": "return=representation"},
            expected=(201,),
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise HostedStorageUnavailable("Supabase returned invalid hosted data.") from exc
        if isinstance(data, list) and data:
            return dict(data[0])
        if isinstance(data, dict):
            return dict(data)
        raise HostedStorageUnavailable("Supabase returned no hosted record.")

    def upsert_row(self, table: str, row: dict[str, Any], *, conflict: str) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"/rest/v1/{table}",
            params={"on_conflict": conflict},
            payload=row,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            # PostgREST commonly returns 201 for a newly inserted upsert and
            # 200 when the conflict path merges an existing row.
            expected=(200, 201),
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise HostedStorageUnavailable("Supabase returned invalid hosted data.") from exc
        if isinstance(data, list) and data:
            return dict(data[0])
        if isinstance(data, dict):
            return dict(data)
        raise HostedStorageUnavailable("Supabase returned no hosted record.")

    def update_rows(
        self, table: str, *, filters: dict[str, str], values: dict[str, Any]
    ) -> list[dict[str, Any]]:
        response = self._request(
            "PATCH",
            f"/rest/v1/{table}",
            params=filters,
            payload=values,
            headers={"Prefer": "return=representation"},
            expected=(200, 204),
        )
        if response.status_code == 204:
            return []
        try:
            data = response.json()
        except ValueError as exc:
            raise HostedStorageUnavailable("Supabase returned invalid hosted data.") from exc
        return [dict(item) for item in data] if isinstance(data, list) else []

    def delete_rows(self, table: str, *, filters: dict[str, str]) -> None:
        self._request("DELETE", f"/rest/v1/{table}", params=filters, expected=(204,))

    def upload_object(
        self,
        *,
        organization_id: str,
        object_id: str,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "file"
        path = f"theriad/{organization_id}/{object_id}-{safe_name}"
        self._request_bytes(
            "POST",
            f"/storage/v1/object/{quote(self.config.bucket, safe='')}/{quote(path, safe='/')}",
            content=content,
            headers={
                "Content-Type": content_type or "application/octet-stream",
                "x-upsert": "false",
            },
            expected=(200, 201),
        )
        return path

    def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        content: bytes,
        headers: dict[str, str],
        expected: tuple[int, ...],
    ) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self.config.url}{path}",
                content=content,
                headers={**self._headers, **headers},
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            raise HostedStorageUnavailable("Supabase hosted storage is unavailable.") from exc
        if response.status_code not in expected:
            raise HostedStorageUnavailable("Supabase hosted file storage returned an error.")
        return response

    def delete_object(self, path: str) -> None:
        if not path.startswith("theriad/"):
            raise HostedStorageUnavailable("Hosted object path is outside the Theriad bucket.")
        self._request(
            "DELETE",
            f"/storage/v1/object/{quote(self.config.bucket, safe='')}/{quote(path, safe='/')}",
            expected=(200, 204),
        )

    def download_object(self, path: str) -> bytes:
        if not path.startswith("theriad/"):
            raise HostedStorageUnavailable("Hosted object path is outside the Theriad bucket.")
        response = self._request("GET", f"/storage/v1/object/{quote(self.config.bucket, safe='')}/{quote(path, safe='/')}", expected=(200,))
        return response.content


def _scope_filters(*, organization_id: str | None, user_id: str | None) -> dict[str, str]:
    filters: dict[str, str] = {}
    if organization_id:
        filters["organization_id"] = f"eq.{organization_id}"
    if user_id:
        filters["user_id"] = f"eq.{user_id}"
    return filters


class HostedRepository:
    """Tenant-scoped repository for one Supabase table."""

    table: str

    def __init__(self, client: SupabaseRestClient | None = None):
        self.client = client or SupabaseRestClient()

    def list_for_owner(
        self,
        *,
        organization_id: str | None,
        user_id: str | None,
        select: str = "*",
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.client.list_rows(
            self.table,
            filters=_scope_filters(organization_id=organization_id, user_id=user_id),
            select=select,
            order=order,
            limit=limit,
        )

    def get_owned(
        self, record_id: str, *, organization_id: str | None, user_id: str | None
    ) -> dict[str, Any] | None:
        rows = self.client.list_rows(
            self.table,
            filters={
                "id": f"eq.{record_id}",
                **_scope_filters(organization_id=organization_id, user_id=user_id),
            },
            limit=1,
        )
        return rows[0] if rows else None

    def insert_owned(
        self,
        row: dict[str, Any],
        *,
        organization_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        owned = {**row, "organization_id": organization_id, "user_id": user_id}
        return self.client.insert_row(self.table, owned)

    def update_owned(
        self,
        record_id: str,
        values: dict[str, Any],
        *,
        organization_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        return self.client.update_rows(
            self.table,
            filters={
                "id": f"eq.{record_id}",
                **_scope_filters(organization_id=organization_id, user_id=user_id),
            },
            values=values,
        )

    def delete_owned(
        self, record_id: str, *, organization_id: str, user_id: str
    ) -> None:
        self.client.delete_rows(
            self.table,
            filters={
                "id": f"eq.{record_id}",
                **_scope_filters(organization_id=organization_id, user_id=user_id),
            },
        )


class OrganizationRepository(HostedRepository):
    table = "organizations"


class MembershipRepository(HostedRepository):
    table = "organization_memberships"


class ProjectRepository(HostedRepository):
    table = "projects"


class ProjectSettingsRepository(HostedRepository):
    table = "project_settings"


class ConversationRepository(HostedRepository):
    table = "conversations"


class MessageRepository(HostedRepository):
    table = "messages"


class FileRepository(HostedRepository):
    table = "project_files"


class RunRepository(HostedRepository):
    table = "runs"


class MemoryRepository(HostedRepository):
    table = "memory_entries"


class MemoryProfileRepository(HostedRepository):
    table = "memory_profiles"


class UsageEventRepository(HostedRepository):
    table = "usage_events"


class AuditEventRepository(HostedRepository):
    table = "audit_events"


class ProviderMetadataRepository(HostedRepository):
    table = "provider_connections"


class BillingCustomerRepository(HostedRepository):
    table = "billing_customers"


class BillingSubscriptionRepository(HostedRepository):
    table = "billing_subscriptions"


class EntitlementRepository(HostedRepository):
    table = "organization_entitlements"


def hosted_repositories(client: SupabaseRestClient | None = None) -> dict[str, HostedRepository]:
    """Construct all named hosted repositories from one transport boundary."""
    client = client or SupabaseRestClient()
    return {
        "organizations": OrganizationRepository(client),
        "memberships": MembershipRepository(client),
        "projects": ProjectRepository(client),
        "project_settings": ProjectSettingsRepository(client),
        "conversations": ConversationRepository(client),
        "messages": MessageRepository(client),
        "files": FileRepository(client),
        "runs": RunRepository(client),
        "memory": MemoryRepository(client),
        "memory_profiles": MemoryProfileRepository(client),
        "usage_events": UsageEventRepository(client),
        "audit_events": AuditEventRepository(client),
        "provider_metadata": ProviderMetadataRepository(client),
        "billing_customers": BillingCustomerRepository(client),
        "billing_subscriptions": BillingSubscriptionRepository(client),
        "entitlements": EntitlementRepository(client),
    }
