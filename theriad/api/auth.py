"""
Lightweight per-user auth for the Theriad workspace.

Every visitor gets their own isolated data scope:

    POST /api/auth/guest     Create a "ghost" free-plan account + session token.
                             Guests can use free chat models, then register to
                             unlock the rest of the workspace.
    POST /api/auth/register  Create a named account. When called with a guest
                             session token, the guest's data (conversations,
                             projects, files, runs) migrates to the new account.
    POST /api/auth/login     Exchange username/password for a session token.
    POST /api/auth/recovery/request
                             Email a one-time verification key for a local
                             account, without disclosing whether it exists.
    POST /api/auth/recovery/complete
                             Replace a local password with the emailed key.
    POST /api/auth/recovery/key, /username, /password
                             Legacy compatibility endpoints for accounts that
                             still have an old recovery key.
    POST /api/auth/supabase/email/request
                             Send a short-lived email code after Google sign-in.
    POST /api/auth/supabase/email/verify
                             Verify the email code and issue a Theriad-scoped
                             session token.
    POST /api/auth/supabase/exchange
                             Legacy TOTP/AAL2 bridge for existing clients.
    POST /api/auth/logout    Invalidate the current session token.
    GET  /api/auth/me        Current user identity.

Tokens are opaque bearer tokens (Authorization: Bearer <token>) stored in the
auth_sessions table. Requests without a token fall back to the legacy
"local" scope (user_id IS NULL) only when ``THERIAD_DEPLOYMENT_MODE=local``.
Hosted mode rejects those requests before private route handlers run, and
token-holding users never see each other's rows.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import smtplib
import ssl
import string
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from theriad.settings.deployment import get_deployment_mode
from theriad.storage.database import get_db
from theriad.storage.organizations import ensure_personal_organization

auth_router = APIRouter(prefix="/api/auth")

_PBKDF2_ITERATIONS = 200_000
_RECOVERY_KEY_PREFIX = "THR-"
_EMAIL_RECOVERY_TTL = timedelta(minutes=15)
_EMAIL_RECOVERY_COOLDOWN = timedelta(minutes=1)
_EMAIL_MFA_TTL = timedelta(minutes=10)
_EMAIL_MFA_COOLDOWN = timedelta(minutes=1)
_EMAIL_MFA_MAX_ATTEMPTS = 5
_GENERIC_RECOVERY_MESSAGE = (
    "If that account has a recovery email, we sent a verification key. "
    "Check your inbox and spam folder."
)

# Public deployments must opt into the hosted boundary explicitly. Keeping
# local as the default preserves the CLI/test contract while making the
# deployment choice visible and reversible during migration.
_DEPLOYMENT_MODE_LOCAL = "local"
_DEPLOYMENT_MODE_HOSTED = "hosted"
_PUBLIC_API_PATHS = frozenset({
    "/api/health",
    "/api/models",
    "/api/pricing",
    "/api/pricing/plans",
    "/api/analytics/collect",
    "/api/billing/webhook",
})
_request_scope: ContextVar[str | None] = ContextVar("theriad_request_scope", default=None)


def deployment_mode() -> str:
    """Return the configured deployment mode, failing closed on bad input."""
    value = (os.getenv("THERIAD_DEPLOYMENT_MODE") or _DEPLOYMENT_MODE_LOCAL).strip().lower()
    return value if value in {_DEPLOYMENT_MODE_LOCAL, _DEPLOYMENT_MODE_HOSTED} else _DEPLOYMENT_MODE_HOSTED


def is_hosted_deployment() -> bool:
    return deployment_mode() == _DEPLOYMENT_MODE_HOSTED


def is_public_api_path(path: str) -> bool:
    """Return whether an API path can be reached without a workspace session."""
    return path in _PUBLIC_API_PATHS


def set_request_scope(scope_id: str | None) -> Token:
    """Bind the authenticated account to downstream provider/storage calls."""
    return _request_scope.set(scope_id)


def reset_request_scope(token: Token) -> None:
    _request_scope.reset(token)


def current_scope_id() -> str | None:
    """Return the account scope bound by the request middleware, if any."""
    return _request_scope.get()


class RecoveryEmailDeliveryError(RuntimeError):
    """Raised when a recovery email cannot be safely delivered."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"{salt}${digest.hex()}"


def _verify_password(password: str, stored: str | None) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, expected = stored.split("$", 1)
    candidate = _hash_password(password, salt).split("$", 1)[1]
    return hmac.compare_digest(candidate, expected)


def _new_recovery_key() -> str:
    """Create a printable, high-entropy recovery key shown only once."""
    raw = secrets.token_hex(16).upper()
    return _RECOVERY_KEY_PREFIX + "-".join(raw[index:index + 4] for index in range(0, len(raw), 4))


def _recovery_key_hash(recovery_key: str) -> str | None:
    """Normalize a copied key before hashing it for constant-shape lookup."""
    normalized = "".join(recovery_key.upper().split()).replace("-", "")
    prefix = _RECOVERY_KEY_PREFIX.replace("-", "")
    if not normalized.startswith(prefix):
        return None
    raw = normalized[len(prefix):]
    if len(raw) != 32 or any(character not in string.hexdigits for character in raw):
        return None
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _new_email_mfa_code() -> str:
    """Create a six-digit code that is delivered once and never logged."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _email_mfa_code_hash(challenge_id: str, code: str) -> str:
    """Bind the short-lived code hash to one challenge record."""
    value = f"{challenge_id}:{code}".encode()
    return hashlib.sha256(value).hexdigest()


def _mask_email(email: str) -> str:
    """Show enough of the destination to help the user without echoing it."""
    local, _, domain = email.partition("@")
    if not local or not domain:
        return "your email address"
    visible = local[:1]
    return f"{visible}{'*' * max(2, min(len(local) - 1, 5))}@{domain}"


def _parse_utc(value: str | None) -> datetime | None:
    """Parse a stored timestamp and normalize legacy naive values to UTC."""
    try:
        parsed = datetime.fromisoformat(value or "")
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _normalize_email(value: str | None) -> str | None:
    """Accept a pragmatic email shape without adding a runtime dependency."""
    if value is None:
        return None
    email = value.strip().lower()
    if not email:
        return None
    local, separator, domain = email.rpartition("@")
    if (
        not separator
        or not local
        or not domain
        or "@" in local
        or "@" in domain
        or len(email) > 254
        or any(character.isspace() for character in email)
        or any(character in email for character in "\r\n")
    ):
        raise HTTPException(status_code=422, detail="Enter a valid recovery email address.")
    return email


def _smtp_settings() -> tuple[str, int, str, str, str, bool]:
    """Return configured SMTP settings without ever exposing their values."""
    host = (os.getenv("THERIAD_SMTP_HOST") or "").strip()
    username = (os.getenv("THERIAD_SMTP_USERNAME") or "").strip()
    password = os.getenv("THERIAD_SMTP_PASSWORD") or ""
    sender = (os.getenv("THERIAD_SMTP_FROM") or username).strip()
    use_tls = (os.getenv("THERIAD_SMTP_USE_TLS") or "true").strip().lower() not in {
        "0", "false", "no", "off",
    }
    try:
        port = int((os.getenv("THERIAD_SMTP_PORT") or "587").strip())
    except ValueError as exc:
        raise RecoveryEmailDeliveryError("Recovery email is not configured.") from exc
    if not host or not sender or not 1 <= port <= 65535 or bool(username) != bool(password):
        raise RecoveryEmailDeliveryError("Recovery email is not configured.")
    return host, port, username, password, sender, use_tls


def _send_smtp_message(message: EmailMessage) -> None:
    """Send one message through the server's configured SMTP relay."""
    host, port, username_for_login, password, sender, use_tls = _smtp_settings()
    if "From" not in message:
        message["From"] = sender
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=10, context=ssl.create_default_context()) as client:
                if username_for_login:
                    client.login(username_for_login, password)
                client.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=10) as client:
                if use_tls:
                    client.starttls(context=ssl.create_default_context())
                if username_for_login:
                    client.login(username_for_login, password)
                client.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise RecoveryEmailDeliveryError("Recovery email could not be delivered.") from exc


def _send_recovery_email(*, recipient: str, username: str, recovery_key: str) -> None:
    """Email a one-time recovery key through the server's configured SMTP relay."""
    app_url = (os.getenv("THERIAD_APP_URL") or "http://localhost:5173").rstrip("/")
    reset_url = f"{app_url}/#/app?{urlencode({'recovery_key': recovery_key})}"
    minutes = int(_EMAIL_RECOVERY_TTL.total_seconds() // 60)

    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = "Your Theriad password reset key"
    message.set_content(
        f"""We received a request to reset the password for your Theriad account.

Username: {username}

Verification key: {recovery_key}

Enter this key in Theriad, or open this secure reset link:
{reset_url}

The key expires in {minutes} minutes and can be used once. If you did not request it, you can ignore this email.
"""
    )
    _send_smtp_message(message)


def _send_email_mfa_code(*, recipient: str, code: str) -> None:
    """Email the sign-in code without exposing it to the browser or logs."""
    minutes = int(_EMAIL_MFA_TTL.total_seconds() // 60)
    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = "Your Theriad sign-in code"
    message.set_content(
        f"""Use this code to finish signing in to Theriad with Google:

{code}

This code expires in {minutes} minutes and can be used once. If you did not try to sign in, you can ignore this email and secure your Google account.
"""
    )
    _send_smtp_message(message)


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        return token or None
    return None


def _ensure_user_organization(user: dict) -> dict:
    """Attach the canonical personal organization to a registered account."""
    if user.get("is_guest"):
        return user
    membership = ensure_personal_organization(
        user["id"], user.get("display_name") or user.get("username")
    )
    user["organization_id"] = membership["organization_id"]
    user["organization_role"] = membership.get("role") or "owner"
    return user


def get_request_user(request: Request) -> dict | None:
    """Return the user row for the request's bearer token, or None."""
    token = _bearer_token(request)
    if not token:
        return None
    db = get_db()
    row = db.execute(
        "SELECT u.*, s.mfa_verified AS session_mfa_verified "
        "FROM auth_sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token = ?",
        (token,),
    ).fetchone()
    if not row:
        return None
    user = dict(row)
    # Google sessions are only issued after the browser has completed the
    # server-side email challenge (or the legacy AAL2 bridge). Keep that fact
    # on the local session so an incomplete Google session cannot silently
    # become an application session.
    if user.get("auth_provider") == "google" and not user.get("session_mfa_verified"):
        return None
    return _ensure_user_organization(user)


def get_auth_context(request: Request) -> dict | None:
    """Return the stable principal/workspace context for a request."""
    user = get_request_user(request)
    if user is None:
        return None
    return {
        "principal_type": "guest" if user.get("is_guest") else "registered",
        "user_id": user["id"],
        "organization_id": user.get("organization_id"),
        "role": user.get("organization_role") or "owner",
        "deployment_mode": deployment_mode(),
    }


def user_scope_id(request: Request) -> str | None:
    """
    The user_id used to scope database queries for this request.
    None means the legacy/local scope (rows with user_id IS NULL).
    """
    if _bearer_token(request) and get_request_user(request) is None:
        raise HTTPException(status_code=401, detail="Your authentication session is invalid or incomplete.")
    user = get_request_user(request)
    return user["id"] if user else None


def _create_session(user_id: str, *, mfa_verified: bool = True) -> str:
    db = get_db()
    token = secrets.token_urlsafe(32)
    db.execute(
        "INSERT INTO auth_sessions (token, user_id, mfa_verified, created_at) "
        "VALUES (?, ?, ?, ?)",
        (token, user_id, int(mfa_verified), _now()),
    )
    db.commit()
    return token


def _user_payload(user: dict) -> dict:
    is_guest = bool(user.get("is_guest"))
    membership = None
    if not is_guest:
        _require_hosted_storage_for_account()
        membership = ensure_personal_organization(
            user["id"], user.get("display_name") or user.get("username")
        )
    organization_id = membership["organization_id"] if membership else user.get("organization_id")
    organization_role = membership.get("role") if membership else user.get("organization_role")
    payload = {
        "id": user["id"],
        "username": user.get("username"),
        "is_guest": is_guest,
        "principal_type": "guest" if is_guest else "registered",
        "organization_id": organization_id,
        "role": organization_role,
        "deployment_mode": get_deployment_mode(),
        "email": user.get("email"),
        "display_name": user.get("display_name"),
        "avatar_url": user.get("avatar_url"),
        "auth_provider": user.get("auth_provider"),
        "supabase_user_id": user.get("supabase_user_id"),
        "mfa_verified": bool(user.get("session_mfa_verified", True)),
        "organization_role": organization_role or ("owner" if not is_guest else None),
    }
    return payload


def _require_hosted_storage_for_account() -> None:
    """Return only when a hosted account can reach its authoritative data plane."""
    from theriad.storage.hosted import (
        HostedStorageUnavailable,
        assert_hosted_storage_ready,
        hosted_data_plane_enabled,
    )

    if not hosted_data_plane_enabled():
        return
    try:
        assert_hosted_storage_ready(probe=True)
    except HostedStorageUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Hosted account storage is unavailable; try again after the service is restored.",
        ) from exc


# ── Endpoints ────────────────────────────────────────────────────────────────

class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=256)
    email: str | None = Field(default=None, max_length=254)


class RecoveryKeyRequest(BaseModel):
    recovery_key: str = Field(min_length=20, max_length=64)


class PasswordRecovery(RecoveryKeyRequest):
    password: str = Field(min_length=6, max_length=256)


class EmailRecoveryRequest(BaseModel):
    identifier: str = Field(min_length=3, max_length=254)


class SupabaseExchange(BaseModel):
    access_token: str = Field(min_length=20, max_length=8192)


class SupabaseEmailChallengeRequest(BaseModel):
    access_token: str = Field(min_length=20, max_length=8192)


class SupabaseEmailChallengeVerification(BaseModel):
    access_token: str = Field(min_length=20, max_length=8192)
    challenge_id: str = Field(min_length=36, max_length=36)
    code: str = Field(min_length=6, max_length=6)


def _supabase_config() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    return url, key


def _jwt_claim(token: str, name: str):
    """Read a claim after Supabase Auth has validated the bearer token."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode("ascii"))).get(name)
    except (IndexError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _verify_supabase_google_token(access_token: str, *, require_aal2: bool = True) -> dict:
    url, key = _supabase_config()
    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail="Supabase authentication is not configured on the Theriad server.",
        )
    try:
        response = httpx.get(
            f"{url}/auth/v1/user",
            headers={"apikey": key, "Authorization": f"Bearer {access_token}"},
            timeout=8.0,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Supabase authentication is unavailable.") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Supabase session is invalid or expired.")
    try:
        user = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Supabase returned an invalid user response.") from exc

    metadata = user.get("app_metadata") or {}
    providers = set(metadata.get("providers") or [])
    if metadata.get("provider"):
        providers.add(metadata["provider"])
    if "google" not in providers:
        raise HTTPException(status_code=403, detail="This endpoint only accepts Google sign-in sessions.")
    if require_aal2 and _jwt_claim(access_token, "aal") != "aal2":
        raise HTTPException(
            status_code=403,
            detail="Two-factor authentication must be verified before Theriad can create a session.",
        )
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="Supabase session did not include a user id.")
    return user


def _upsert_supabase_user(user: dict, request: Request) -> dict:
    db = get_db()
    metadata = user.get("user_metadata") or {}
    email = user.get("email")
    display_name = metadata.get("full_name") or metadata.get("name") or email or "Theriad user"
    avatar_url = metadata.get("avatar_url") or metadata.get("picture")
    supabase_user_id = user["id"]
    username = email or display_name
    current = get_request_user(request)
    existing = db.execute(
        "SELECT * FROM users WHERE supabase_user_id = ?", (supabase_user_id,)
    ).fetchone()
    if current and current.get("is_guest") and not existing:
        user_id = current["id"]
        db.execute(
            "UPDATE users SET username = COALESCE(username, ?), email = ?, "
            "display_name = ?, avatar_url = ?, auth_provider = 'google', "
            "supabase_user_id = ?, is_guest = 0 WHERE id = ?",
            (email or display_name, email, display_name, avatar_url, supabase_user_id, user_id),
        )
    elif existing:
        user_id = existing["id"]
        db.execute(
            "UPDATE users SET email = ?, display_name = ?, avatar_url = ?, "
            "auth_provider = 'google', is_guest = 0 WHERE id = ?",
            (email, display_name, avatar_url, user_id),
        )
    else:
        username_taken = db.execute(
            "SELECT id FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()
        if username_taken:
            username = f"{display_name}-{supabase_user_id[:8]}"
        user_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO users (id, username, password_hash, is_guest, email, "
            "display_name, avatar_url, auth_provider, supabase_user_id, created_at) "
            "VALUES (?, ?, NULL, 0, ?, ?, ?, 'google', ?, ?)",
            (user_id, username, email, display_name, avatar_url, supabase_user_id, _now()),
        )
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _ensure_user_organization(dict(row))


@auth_router.post("/guest", status_code=201)
def create_guest_session():
    """Create a ghost free-plan account and return its session token."""
    db = get_db()
    user_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO users (id, username, password_hash, is_guest, created_at) "
        "VALUES (?, NULL, NULL, 1, ?)",
        (user_id, _now()),
    )
    db.commit()
    token = _create_session(user_id)
    user = _ensure_user_organization(
        dict(db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
    )
    return {"token": token, "user": _user_payload(user)}


@auth_router.post("/register", status_code=201)
def register(body: Credentials, request: Request):
    """
    Create a named account. If the caller holds a guest session, the guest's
    data (conversations, projects, files, runs) is migrated to the account.
    """
    db = get_db()
    username = body.username.strip()
    email = _normalize_email(body.email)
    existing = db.execute(
        "SELECT id FROM users WHERE username = ? COLLATE NOCASE", (username,)
    ).fetchone()
    if existing:
        raise HTTPException(status_code=409, detail="That username is already taken.")

    # Do this before mutating the local auth database. Hosted mode must not
    # create an account that cannot be backed by the authoritative Supabase
    # organization and migration records.
    _require_hosted_storage_for_account()

    current = get_request_user(request)
    now = _now()

    if current and current.get("is_guest"):
        # Upgrade the ghost account in place so its data comes along.
        user_id = current["id"]
        db.execute(
            "UPDATE users SET username = ?, password_hash = ?, email = COALESCE(?, email), "
            "recovery_key_hash = NULL, is_guest = 0 "
            "WHERE id = ?",
            (username, _hash_password(body.password), email, user_id),
        )
    else:
        user_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO users (id, username, password_hash, is_guest, email, created_at) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (user_id, username, _hash_password(body.password), email, now),
        )
    db.commit()
    token = _create_session(user_id)
    user = dict(db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
    migration = {"status": "not_required"}
    try:
        from theriad.storage.hosted import hosted_data_plane_enabled
        from theriad.storage.migration import migrate_user_workspace_to_hosted

        if hosted_data_plane_enabled():
            membership = ensure_personal_organization(user_id, username)
            migration = migrate_user_workspace_to_hosted(
                user_id=user_id,
                organization_id=membership["organization_id"],
                organization_name=membership["name"],
            )
    except Exception:
        # Registration remains usable while the operator repairs a hosted
        # dependency; the response tells the client that migration is pending
        # without surfacing provider or database details.
        migration = {"status": "pending"}
    return {
        "token": token,
        "user": _user_payload(user),
        "hosted_migration": migration,
    }


@auth_router.post("/login")
def login(body: Credentials):
    db = get_db()
    row = db.execute(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE AND is_guest = 0",
        (body.username.strip(),),
    ).fetchone()
    if row is None or not _verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = _create_session(row["id"], mfa_verified=True)
    return {"token": token, "user": _user_payload(dict(row))}


def _email_recovery_user(recovery_key: str) -> dict:
    key_hash = _recovery_key_hash(recovery_key)
    if not key_hash:
        raise HTTPException(status_code=400, detail="That verification key was not recognized. Check it and try again.")
    db = get_db()
    row = db.execute(
        "SELECT u.*, r.id AS recovery_token_id FROM auth_recovery_tokens r "
        "JOIN users u ON u.id = r.user_id "
        "WHERE r.token_hash = ? AND r.used_at IS NULL AND r.expires_at > ? "
        "AND u.is_guest = 0 AND u.password_hash IS NOT NULL",
        (key_hash, _now()),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="That verification key was not recognized. Check it and try again.")
    return dict(row)


@auth_router.post("/recovery/request")
def request_email_recovery(body: EmailRecoveryRequest):
    """Send one recovery key at most once per minute without account enumeration."""
    try:
        _smtp_settings()
    except RecoveryEmailDeliveryError as exc:
        # This is a deployment-wide state, so it is safe to state plainly and
        # cannot reveal whether the submitted account exists.
        raise HTTPException(
            status_code=503,
            detail="Email recovery is not configured on this Theriad server.",
        ) from exc
    identifier = body.identifier.strip()
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE is_guest = 0 AND "
        "(username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE) LIMIT 1",
        (identifier, identifier),
    ).fetchone()
    if not user or not user["email"] or not user["password_hash"] or user["auth_provider"] == "google":
        return {"message": _GENERIC_RECOVERY_MESSAGE}

    now = datetime.now(timezone.utc)
    latest = db.execute(
        "SELECT created_at FROM auth_recovery_tokens WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (user["id"],),
    ).fetchone()
    if latest:
        try:
            last_sent = datetime.fromisoformat(latest["created_at"])
        except (TypeError, ValueError):
            last_sent = None
        if last_sent and last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=timezone.utc)
        if last_sent and last_sent > now - _EMAIL_RECOVERY_COOLDOWN:
            return {"message": _GENERIC_RECOVERY_MESSAGE}

    recovery_key = _new_recovery_key()
    token_hash = _recovery_key_hash(recovery_key)
    if token_hash is None:  # pragma: no cover - generated keys always have this shape
        raise HTTPException(status_code=500, detail="Unable to create a verification key.")
    token_id = str(uuid.uuid4())
    now_text = now.isoformat()
    expires_at = (now + _EMAIL_RECOVERY_TTL).isoformat()
    # A new request invalidates any unconsumed key for the same account.
    db.execute(
        "UPDATE auth_recovery_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL",
        (now_text, user["id"]),
    )
    db.execute(
        "INSERT INTO auth_recovery_tokens (id, user_id, token_hash, expires_at, used_at, created_at) "
        "VALUES (?, ?, ?, ?, NULL, ?)",
        (token_id, user["id"], token_hash, expires_at, now_text),
    )
    db.commit()
    try:
        _send_recovery_email(
            recipient=user["email"],
            username=user["username"],
            recovery_key=recovery_key,
        )
    except RecoveryEmailDeliveryError:
        # Never leave an unreceived, usable key behind if the relay fails.
        db.execute("DELETE FROM auth_recovery_tokens WHERE id = ?", (token_id,))
        db.commit()
        return {"message": _GENERIC_RECOVERY_MESSAGE}
    return {"message": _GENERIC_RECOVERY_MESSAGE}


@auth_router.post("/recovery/complete")
def complete_email_recovery(body: PasswordRecovery):
    """Consume an emailed verification key, reset the password, and sign in."""
    user = _email_recovery_user(body.recovery_key)
    db = get_db()
    now = _now()
    consumed = db.execute(
        "UPDATE auth_recovery_tokens SET used_at = ? "
        "WHERE id = ? AND used_at IS NULL",
        (now, user["recovery_token_id"]),
    )
    if consumed.rowcount != 1:
        raise HTTPException(status_code=400, detail="That verification key was already used. Request a new one.")
    db.execute(
        "UPDATE users SET password_hash = ?, recovery_key_hash = NULL WHERE id = ?",
        (_hash_password(body.password), user["id"]),
    )
    # A password reset signs out every browser holding an old session.
    db.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user["id"],))
    db.commit()
    token = _create_session(user["id"])
    refreshed = dict(db.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone())
    return {"token": token, "user": _user_payload(refreshed)}


def _user_for_recovery_key(recovery_key: str) -> dict:
    key_hash = _recovery_key_hash(recovery_key)
    if not key_hash:
        raise HTTPException(status_code=400, detail="That recovery key was not recognized. Check it and try again.")
    db = get_db()
    row = db.execute(
        "SELECT * FROM users WHERE recovery_key_hash = ? AND is_guest = 0",
        (key_hash,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="That recovery key was not recognized. Check it and try again.")
    return dict(row)


@auth_router.post("/recovery/key")
def issue_recovery_key(request: Request):
    """Issue a recovery key while the account is already authenticated."""
    user = get_request_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in before creating a recovery key.")
    if user.get("is_guest"):
        raise HTTPException(status_code=400, detail="Create a named account before creating a recovery key.")
    recovery_key = _new_recovery_key()
    db = get_db()
    db.execute(
        "UPDATE users SET recovery_key_hash = ? WHERE id = ?",
        (_recovery_key_hash(recovery_key), user["id"]),
    )
    db.commit()
    return {"recovery_key": recovery_key}


@auth_router.post("/recovery/username")
def recover_username(body: RecoveryKeyRequest):
    """Recover a local username without exposing an account directory."""
    user = _user_for_recovery_key(body.recovery_key)
    return {"username": user.get("username")}


@auth_router.post("/recovery/password")
def recover_password(body: PasswordRecovery):
    """Replace a local password and rotate the one-time-displayed key."""
    user = _user_for_recovery_key(body.recovery_key)
    recovery_key = _new_recovery_key()
    db = get_db()
    db.execute(
        "UPDATE users SET password_hash = ?, recovery_key_hash = ? WHERE id = ?",
        (_hash_password(body.password), _recovery_key_hash(recovery_key), user["id"]),
    )
    # A password reset signs out every other browser holding the old session.
    db.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user["id"],))
    db.commit()
    token = _create_session(user["id"])
    refreshed = dict(db.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone())
    return {
        "token": token,
        "user": _user_payload(refreshed),
        "recovery_key": recovery_key,
    }


@auth_router.post("/supabase/email/request")
def request_supabase_email_code(body: SupabaseEmailChallengeRequest):
    """Send a one-time email code for a validated Google session."""
    user = _verify_supabase_google_token(body.access_token, require_aal2=False)
    email = (user.get("email") or "").strip().lower()
    if (
        not email
        or "@" not in email
        or any(character in email for character in "\r\n")
    ):
        raise HTTPException(status_code=403, detail="The Google account does not have a usable email address.")

    db = get_db()
    now = datetime.now(timezone.utc)
    latest = db.execute(
        "SELECT id, email, expires_at, created_at FROM auth_email_mfa_challenges "
        "WHERE supabase_user_id = ? AND used_at IS NULL "
        "ORDER BY created_at DESC LIMIT 1",
        (user["id"],),
    ).fetchone()
    if latest:
        created_at = _parse_utc(latest["created_at"])
        expires_at = _parse_utc(latest["expires_at"])
        if (
            latest["email"] == email
            and created_at
            and expires_at
            and created_at > now - _EMAIL_MFA_COOLDOWN
            and expires_at > now
        ):
            return {
                "challenge_id": latest["id"],
                "email": _mask_email(email),
                "expires_in": max(1, int((expires_at - now).total_seconds())),
            }

    challenge_id = str(uuid.uuid4())
    code = _new_email_mfa_code()
    created_at = now.isoformat()
    expires_at = (now + _EMAIL_MFA_TTL).isoformat()
    db.execute(
        "UPDATE auth_email_mfa_challenges SET used_at = ? "
        "WHERE supabase_user_id = ? AND used_at IS NULL",
        (created_at, user["id"]),
    )
    db.execute(
        "INSERT INTO auth_email_mfa_challenges "
        "(id, supabase_user_id, email, code_hash, expires_at, used_at, attempts, created_at) "
        "VALUES (?, ?, ?, ?, ?, NULL, 0, ?)",
        (
            challenge_id,
            user["id"],
            email,
            _email_mfa_code_hash(challenge_id, code),
            expires_at,
            created_at,
        ),
    )
    db.commit()
    try:
        _send_email_mfa_code(recipient=email, code=code)
    except RecoveryEmailDeliveryError as exc:
        # Do not leave a usable code behind when delivery failed.
        db.execute("DELETE FROM auth_email_mfa_challenges WHERE id = ?", (challenge_id,))
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="Email sign-in is not configured or unavailable on this Theriad server.",
        ) from exc
    return {
        "challenge_id": challenge_id,
        "email": _mask_email(email),
        "expires_in": int(_EMAIL_MFA_TTL.total_seconds()),
    }


@auth_router.post("/supabase/email/verify")
def verify_supabase_email_code(body: SupabaseEmailChallengeVerification, request: Request):
    """Consume the email code and create the scoped Theriad session."""
    if not body.code.isdigit():
        raise HTTPException(status_code=422, detail="Enter the six-digit code from your email.")
    user = _verify_supabase_google_token(body.access_token, require_aal2=False)
    email = (user.get("email") or "").strip().lower()
    db = get_db()
    row = db.execute(
        "SELECT * FROM auth_email_mfa_challenges "
        "WHERE id = ? AND supabase_user_id = ? AND email = ? AND used_at IS NULL",
        (body.challenge_id, user["id"], email),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="That email code was not recognized. Request a new code.")

    now = datetime.now(timezone.utc)
    expires_at = _parse_utc(row["expires_at"])
    if not expires_at or expires_at <= now:
        raise HTTPException(status_code=400, detail="That email code has expired. Request a new code.")
    if row["attempts"] >= _EMAIL_MFA_MAX_ATTEMPTS:
        raise HTTPException(status_code=400, detail="Too many incorrect attempts. Request a new code.")
    expected_hash = _email_mfa_code_hash(body.challenge_id, body.code)
    if not hmac.compare_digest(expected_hash, row["code_hash"]):
        db.execute(
            "UPDATE auth_email_mfa_challenges SET attempts = attempts + 1 "
            "WHERE id = ? AND used_at IS NULL",
            (body.challenge_id,),
        )
        db.commit()
        raise HTTPException(status_code=400, detail="That email code was not accepted.")

    consumed = db.execute(
        "UPDATE auth_email_mfa_challenges SET used_at = ? "
        "WHERE id = ? AND used_at IS NULL",
        (_now(), body.challenge_id),
    )
    if consumed.rowcount != 1:
        raise HTTPException(status_code=400, detail="That email code was already used. Request a new code.")
    db.commit()

    local_user = _upsert_supabase_user(user, request)
    token = _create_session(local_user["id"], mfa_verified=True)
    local_user["session_mfa_verified"] = 1
    return {
        "token": token,
        "user": _user_payload(local_user),
        "supabase": {
            "id": user["id"],
            "email": user.get("email"),
            "mfa_verified": True,
            "method": "email",
        },
    }


@auth_router.post("/supabase/exchange")
def exchange_supabase_session(body: SupabaseExchange, request: Request):
    """Legacy bridge for clients that still complete Supabase AAL2."""
    user = _verify_supabase_google_token(body.access_token)
    local_user = _upsert_supabase_user(user, request)
    token = _create_session(local_user["id"], mfa_verified=True)
    local_user["session_mfa_verified"] = 1
    migration = {"status": "not_required"}
    try:
        from theriad.storage.hosted import hosted_data_plane_enabled
        from theriad.storage.migration import migrate_user_workspace_to_hosted

        if hosted_data_plane_enabled():
            membership = ensure_personal_organization(
                local_user["id"], local_user.get("display_name") or local_user.get("email")
            )
            migration = migrate_user_workspace_to_hosted(
                user_id=local_user["id"],
                organization_id=membership["organization_id"],
                organization_name=membership["name"],
            )
    except Exception:
        migration = {"status": "pending"}
    return {
        "token": token,
        "user": _user_payload(local_user),
        "supabase": {
            "id": user["id"],
            "email": user.get("email"),
            "mfa_verified": True,
        },
        "hosted_migration": migration,
    }


@auth_router.post("/logout")
def logout(request: Request):
    token = _bearer_token(request)
    if token:
        db = get_db()
        db.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
        db.commit()
    return {"ok": True}


@auth_router.get("/security")
def security_summary(request: Request):
    """Return safe account-security metadata; never return tokens or secrets."""
    user = get_request_user(request)
    token = _bearer_token(request)
    if user is None or not token:
        raise HTTPException(status_code=401, detail="Sign in before viewing account security.")
    db = get_db()
    current = db.execute(
        "SELECT created_at, mfa_verified FROM auth_sessions WHERE token = ? AND user_id = ?",
        (token, user["id"]),
    ).fetchone()
    count = db.execute(
        "SELECT COUNT(*) AS count FROM auth_sessions WHERE user_id = ?",
        (user["id"],),
    ).fetchone()
    provider = user.get("auth_provider") or "local"
    return {
        "auth_provider": provider,
        "mfa_status": "verified" if provider == "google" and bool(user.get("session_mfa_verified")) else "not_configured",
        "recovery_key_available": bool(user.get("recovery_key_hash")),
        "session_count": int(count["count"] or 0),
        "current_session_created_at": current["created_at"] if current else None,
        "current_session_mfa_verified": bool(current["mfa_verified"]) if current else False,
    }


@auth_router.post("/sessions/revoke-others")
def revoke_other_sessions(request: Request):
    """Revoke every session except the bearer token used for this request."""
    user = get_request_user(request)
    token = _bearer_token(request)
    if user is None or not token:
        raise HTTPException(status_code=401, detail="Sign in before revoking sessions.")
    db = get_db()
    result = db.execute(
        "DELETE FROM auth_sessions WHERE user_id = ? AND token != ?",
        (user["id"], token),
    )
    db.commit()
    return {"ok": True, "revoked": int(result.rowcount or 0)}


@auth_router.get("/me")
def me(request: Request):
    user = get_request_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return {"user": _user_payload(user)}
