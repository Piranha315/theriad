"""
SQLite database layer for Theriad project/conversation/file/run persistence.

The database lives at <data_dir>/theriad.db  (default: ./data/theriad.db relative to cwd).
Override by passing db_path to get_db() or by calling init_db(path).

Tables
------
  projects        — named workspaces with optional system instructions
  conversations   — threaded conversations inside a project
  messages        — individual turns (user / assistant) in a conversation
  project_files   — uploaded text/code files attached to a project
  skills          — user-installed behavioral instructions (global or project-scoped)
  runs            — execution records (model, cost, latency, …)

All primary keys are random UUIDs (text).
Timestamps are ISO-8601 strings in UTC.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

# ── Thread-local connection cache ─────────────────────────────────────────────
_local = threading.local()
_db_path: Path | None = None
_schema_initialized: dict[str, bool] = {}  # keyed by db path; prevents repeat migrations
_schema_lock = threading.Lock()


def configure(db_path: str | Path) -> None:
    """Set the database path before the first connection is opened."""
    global _db_path
    _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_path() -> Path:
    if _db_path is not None:
        return _db_path
    default = Path("data") / "theriad.db"
    default.parent.mkdir(parents=True, exist_ok=True)
    return default


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """
    Return a thread-local SQLite connection.

    The first call for a given thread opens the connection and runs init_db().
    Subsequent calls on the same thread reuse the existing connection.
    Pass db_path to override the global path (useful in tests).
    """
    path = Path(db_path) if db_path else _resolve_path()
    key = str(path)

    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    conn_key: str | None = getattr(_local, "conn_key", None)

    if conn is None or conn_key != key:
        if conn is not None:
            conn.close()
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _local.conn_key = key
        # Run schema init/migration only once per process per db path.
        if key not in _schema_initialized:
            with _schema_lock:
                if key not in _schema_initialized:
                    _init_schema(conn)
                    _schema_initialized[key] = True

    return conn


def close_db() -> None:
    """Close the thread-local connection (call in tests / shutdown)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
        _local.conn_key = None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS projects (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        description TEXT,
        instructions TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        project_id  TEXT REFERENCES projects(id) ON DELETE CASCADE,
        title       TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
        id               TEXT PRIMARY KEY,
        conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role             TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content          TEXT NOT NULL,
        created_at       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS project_files (
        id             TEXT PRIMARY KEY,
        project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        filename       TEXT NOT NULL,
        original_name  TEXT NOT NULL,
        mime_type      TEXT,
        size_bytes     INTEGER NOT NULL,
        extracted_text TEXT,
        created_at     TEXT NOT NULL
    );

    -- Imported skills are prompt-time guidance, not model training. A NULL
    -- project_id makes a skill global to this local installation; otherwise it
    -- is available only inside the referenced project. Imports default to
    -- disabled so users can review untrusted instructions before activation.
    CREATE TABLE IF NOT EXISTS skills (
        id             TEXT PRIMARY KEY,
        user_id        TEXT,
        project_id     TEXT REFERENCES projects(id) ON DELETE CASCADE,
        name           TEXT NOT NULL,
        description    TEXT NOT NULL DEFAULT '',
        instructions   TEXT NOT NULL,
        enabled        INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0, 1)),
        priority       INTEGER NOT NULL DEFAULT 100,
        source_type    TEXT NOT NULL DEFAULT 'manual',
        source_name    TEXT,
        content_sha256 TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_skills_effective
        ON skills (enabled, project_id, priority, name);

    CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_scope_content
        ON skills (COALESCE(user_id, ''), COALESCE(project_id, ''), content_sha256);

    CREATE TABLE IF NOT EXISTS runs (
        id                  TEXT PRIMARY KEY,
        project_id          TEXT,
        conversation_id     TEXT,
        message_id          TEXT,
        model               TEXT,
        tier                TEXT,
        provider            TEXT,
        policy              TEXT,
        input_tokens        INTEGER,
        output_tokens       INTEGER,
        actual_cost_usd     REAL,
        routing_latency_ms  REAL,
        provider_latency_ms REAL,
        total_latency_ms    REAL,
        fallback_used       INTEGER DEFAULT 0,
        status              TEXT,
        error               TEXT,
        created_at          TEXT NOT NULL
    );

    -- Privacy-preserving Router V2 counterfactuals and outcomes. Raw prompts
    -- are intentionally absent; prompt_hash is a normalized SHA-256 digest.
    CREATE TABLE IF NOT EXISTS routing_observations (
        id                     TEXT PRIMARY KEY,
        decision_id            TEXT NOT NULL,
        run_id                 TEXT REFERENCES runs(id) ON DELETE SET NULL,
        user_id                TEXT,
        prompt_hash            TEXT NOT NULL,
        prompt_embedding       TEXT,
        router_version         TEXT NOT NULL,
        lane                   TEXT NOT NULL,
        model                  TEXT NOT NULL,
        role                   TEXT NOT NULL,
        propensity             REAL,
        p_success              REAL,
        p_success_lower        REAL,
        expected_output_tokens REAL,
        expected_cost_usd      REAL,
        expected_latency_ms    REAL,
        uncertainty            REAL,
        eligible               INTEGER NOT NULL DEFAULT 1,
        exclusion_reason       TEXT,
        realized_success       REAL,
        realized_score         REAL,
        realized_cost_usd      REAL,
        realized_latency_ms    REAL,
        input_tokens           INTEGER,
        output_tokens          INTEGER,
        evaluator              TEXT,
        evaluator_version      TEXT,
        created_at             TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_routing_observations_decision
        ON routing_observations (decision_id, model);
    CREATE INDEX IF NOT EXISTS idx_routing_observations_router_created
        ON routing_observations (router_version, created_at);

    -- Project-level model filtering and budget settings.
    -- One row per project (created on first PATCH /api/projects/{id}/settings).
    -- JSON-encoded lists are stored as TEXT; NULL means "no restriction".
    -- Local user accounts. Guests (is_guest=1) are auto-created "ghost"
    -- free-plan accounts; registering migrates their rows to the new account.
    CREATE TABLE IF NOT EXISTS users (
        id               TEXT PRIMARY KEY,
        username         TEXT UNIQUE,
        password_hash    TEXT,
        is_guest         INTEGER NOT NULL DEFAULT 0 CHECK(is_guest IN (0, 1)),
        recovery_key_hash TEXT,
        email            TEXT,
        display_name     TEXT,
        avatar_url       TEXT,
        auth_provider    TEXT,
        supabase_user_id TEXT,
        created_at       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS auth_sessions (
        token         TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        mfa_verified  INTEGER NOT NULL DEFAULT 1 CHECK(mfa_verified IN (0, 1)),
        created_at    TEXT NOT NULL
    );

    -- One-time verification codes sent by email for local-account recovery.
    -- Only a SHA-256 hash is stored; expired and used codes cannot reset a
    -- password or disclose whether an account exists.
    CREATE TABLE IF NOT EXISTS auth_recovery_tokens (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash  TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        used_at     TEXT,
        created_at  TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_auth_recovery_tokens_hash
        ON auth_recovery_tokens (token_hash);

    CREATE INDEX IF NOT EXISTS idx_auth_recovery_tokens_user
        ON auth_recovery_tokens (user_id, created_at DESC);

    -- Short-lived email codes used to finish Google sign-in. The Supabase
    -- session is validated separately; this table only holds the one-time
    -- challenge state and a hash of the code sent through SMTP.
    CREATE TABLE IF NOT EXISTS auth_email_mfa_challenges (
        id                TEXT PRIMARY KEY,
        supabase_user_id  TEXT NOT NULL,
        email             TEXT NOT NULL,
        code_hash         TEXT NOT NULL,
        expires_at        TEXT NOT NULL,
        used_at           TEXT,
        attempts          INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_auth_email_mfa_user
        ON auth_email_mfa_challenges (supabase_user_id, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_auth_email_mfa_expiry
        ON auth_email_mfa_challenges (expires_at, used_at);

    -- Every registered account gets one explicit personal organization. The
    -- first tenancy slice keeps existing resources keyed by user_id; these
    -- rows provide the stable owner boundary for a later team-organization
    -- migration without claiming legacy NULL-scoped resources.
    CREATE TABLE IF NOT EXISTS organizations (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        kind        TEXT NOT NULL DEFAULT 'personal'
                    CHECK(kind IN ('personal', 'team')),
        created_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS organization_memberships (
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK(role IN ('owner', 'admin', 'member', 'auditor')),
        personal        INTEGER NOT NULL DEFAULT 0 CHECK(personal IN (0, 1)),
        created_at      TEXT NOT NULL,
        PRIMARY KEY (organization_id, user_id)
    );

    CREATE INDEX IF NOT EXISTS idx_organization_memberships_user
        ON organization_memberships (user_id, created_at);

    CREATE UNIQUE INDEX IF NOT EXISTS idx_one_personal_org_per_user
        ON organization_memberships (user_id) WHERE personal = 1;

    -- Billing state is organization-owned. Stripe is an external source of
    -- truth; these rows are the minimum local projection needed for request
    -- authorization and idempotent webhook handling.
    CREATE TABLE IF NOT EXISTS billing_customers (
        organization_id  TEXT PRIMARY KEY REFERENCES organizations(id) ON DELETE CASCADE,
        stripe_customer_id TEXT NOT NULL UNIQUE,
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS billing_subscriptions (
        id                   TEXT PRIMARY KEY,
        organization_id      TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        stripe_subscription_id TEXT NOT NULL UNIQUE,
        stripe_customer_id   TEXT,
        plan_id              TEXT NOT NULL,
        interval             TEXT,
        status               TEXT NOT NULL,
        current_period_end   TEXT,
        cancel_at_period_end INTEGER NOT NULL DEFAULT 0 CHECK(cancel_at_period_end IN (0, 1)),
        created_at           TEXT NOT NULL,
        updated_at           TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_billing_subscriptions_org
        ON billing_subscriptions (organization_id, updated_at);

    CREATE TABLE IF NOT EXISTS billing_events (
        event_id       TEXT PRIMARY KEY,
        event_type     TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        received_at    TEXT NOT NULL,
        processed_at   TEXT
    );

    CREATE TABLE IF NOT EXISTS organization_entitlements (
        organization_id TEXT PRIMARY KEY REFERENCES organizations(id) ON DELETE CASCADE,
        plan_id         TEXT NOT NULL DEFAULT 'free',
        status          TEXT NOT NULL DEFAULT 'active',
        updated_at      TEXT NOT NULL
    );

    -- Immutable per-run usage and Theriad routing-fee ledger. Provider usage
    -- belongs to the run; the routing fee belongs to this tenant-owned event.
    -- A stable idempotency key prevents retries or duplicate webhook/worker
    -- passes from charging the same run twice.
    CREATE TABLE IF NOT EXISTS usage_events (
        id                    TEXT PRIMARY KEY,
        idempotency_key       TEXT NOT NULL UNIQUE,
        organization_id       TEXT,
        user_id               TEXT,
        run_id                TEXT,
        provider              TEXT,
        model                 TEXT,
        input_tokens          INTEGER,
        output_tokens         INTEGER,
        provider_cost_usd     REAL,
        routing_fee_usd       REAL,
        total_billed_usd      REAL,
        status                TEXT NOT NULL DEFAULT 'pending',
        billing_period        TEXT NOT NULL,
        stripe_invoice_item_id TEXT,
        metadata_json         TEXT NOT NULL DEFAULT '{}',
        created_at            TEXT NOT NULL,
        reported_at           TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_usage_events_owner_period
        ON usage_events (organization_id, user_id, billing_period, created_at);

    CREATE INDEX IF NOT EXISTS idx_usage_events_pending
        ON usage_events (status, billing_period, created_at);

    CREATE TABLE IF NOT EXISTS audit_events (
        id              TEXT PRIMARY KEY,
        organization_id TEXT,
        user_id         TEXT,
        event_type      TEXT NOT NULL,
        resource_type   TEXT,
        resource_id     TEXT,
        metadata_json   TEXT NOT NULL DEFAULT '{}',
        created_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_audit_events_owner_created
        ON audit_events (organization_id, user_id, created_at);

    -- Anonymous, first-party site-traffic measurement. Visitor IDs live in
    -- an HttpOnly cookie; no IP address, user agent, or account identity is
    -- stored. Sessions are kept separate from auth sessions so site analytics
    -- never exposes chat/account data.
    CREATE TABLE IF NOT EXISTS analytics_visitors (
        visitor_id TEXT PRIMARY KEY,
        first_seen TEXT NOT NULL,
        last_seen  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS analytics_sessions (
        session_id      TEXT PRIMARY KEY,
        visitor_id      TEXT NOT NULL REFERENCES analytics_visitors(visitor_id)
                        ON DELETE CASCADE,
        started_at      TEXT NOT NULL,
        last_seen       TEXT NOT NULL,
        ended_at        TEXT,
        landing_path    TEXT NOT NULL,
        landing_referrer TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_analytics_sessions_last_seen
        ON analytics_sessions (last_seen);

    CREATE TABLE IF NOT EXISTS analytics_events (
        id          TEXT PRIMARY KEY,
        visitor_id  TEXT NOT NULL REFERENCES analytics_visitors(visitor_id)
                    ON DELETE CASCADE,
        session_id  TEXT NOT NULL REFERENCES analytics_sessions(session_id)
                    ON DELETE CASCADE,
        event_type  TEXT NOT NULL CHECK(event_type IN ('page_view', 'heartbeat', 'session_end')),
        path        TEXT NOT NULL,
        referrer    TEXT,
        created_at  TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_analytics_events_type_created
        ON analytics_events (event_type, created_at);

    -- Account-scoped recovery grants for the private analytics dashboard.
    -- The passcode is never stored here; only its successful, time-limited
    -- authorization result is retained.
    CREATE TABLE IF NOT EXISTS analytics_access_grants (
        user_id    TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        granted_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_analytics_access_grants_expires
        ON analytics_access_grants (expires_at);

    CREATE TABLE IF NOT EXISTS project_settings (
        project_id              TEXT PRIMARY KEY
                                    REFERENCES projects(id) ON DELETE CASCADE,
        allowed_model_patterns  TEXT,     -- JSON array, e.g. '["qwen/*","ollama/*"]'
        blocked_model_patterns  TEXT,     -- JSON array
        max_cost_per_call_usd   REAL,     -- NULL = no cap
        allow_premium           INTEGER NOT NULL DEFAULT 1,
        local_only              INTEGER NOT NULL DEFAULT 0,
        enable_openrouter_auto_fallback INTEGER NOT NULL DEFAULT 0,
        updated_at              TEXT NOT NULL
    );

    -- Account-scoped runtime controls. The __local__ scope preserves the
    -- unauthenticated local/dev behavior while bearer-authenticated users get
    -- isolated settings keyed by their local Theriad user id.
    CREATE TABLE IF NOT EXISTS runtime_settings (
        scope_id                    TEXT PRIMARY KEY,
        regular_chat_returns_answer INTEGER NOT NULL DEFAULT 1,
        allow_follow_up_questions   INTEGER NOT NULL DEFAULT 1,
        route_only_available        INTEGER NOT NULL DEFAULT 0,
        disable_premium_models      INTEGER NOT NULL DEFAULT 0,
        model_origin_policy         TEXT NOT NULL DEFAULT 'any',
        max_cost_per_request_usd    REAL DEFAULT NULL,
        max_daily_spend_usd         REAL DEFAULT 5.0,
        updated_at                  TEXT NOT NULL
    );
    """)
    conn.commit()

    # ── Additive column migrations for existing databases ─────────────────────
    # SQLite does not support IF NOT EXISTS in ALTER TABLE; use try/except.
    _add_run_columns(conn)
    _migrate_conversations_nullable_project(conn)
    _add_user_scope_columns(conn)
    _migrate_files_nullable_project(conn)
    _add_runtime_settings_columns(conn)
    # Create the Supabase identity index only after additive auth migrations
    # have added its column to pre-Supabase users tables.
    _add_auth_columns(conn)


def _migrate_conversations_nullable_project(conn: sqlite3.Connection) -> None:
    """
    Make conversations.project_id nullable so standalone (projectless) chat
    conversations are possible.  Existing databases may have a NOT NULL
    constraint on that column — this migration recreates the table without it.
    Idempotent: if the column is already nullable the probe insert succeeds and
    the migration is skipped.
    """
    try:
        conn.execute(
            "INSERT INTO conversations (id, project_id, title, created_at, updated_at) "
            "VALUES ('__migration_probe__', NULL, 'probe', 'x', 'x')"
        )
        conn.execute("DELETE FROM conversations WHERE id = '__migration_probe__'")
        conn.commit()
    except sqlite3.IntegrityError:
        # Column has NOT NULL constraint — recreate without it
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations_v2 (
            id          TEXT PRIMARY KEY,
            project_id  TEXT REFERENCES projects(id) ON DELETE CASCADE,
            title       TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        INSERT OR IGNORE INTO conversations_v2
            SELECT id, project_id, title, created_at, updated_at
            FROM conversations;
        DROP TABLE conversations;
        ALTER TABLE conversations_v2 RENAME TO conversations;
        """)
        conn.commit()


def _add_user_scope_columns(conn: sqlite3.Connection) -> None:
    """
    Add user_id ownership columns so each account only sees its own data.
    Legacy rows keep user_id NULL and are only visible to unauthenticated
    (local/dev) requests. Idempotent.
    """
    for table in ("projects", "conversations", "runs", "project_files", "skills"):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
            conn.commit()
        except sqlite3.OperationalError as exc:
            # Only "duplicate column" is expected; anything else (e.g. a
            # transient lock) must surface, or the schema silently diverges.
            if "duplicate column" not in str(exc).lower():
                raise
    # The first version of the skill store used a global uniqueness key. It is
    # still safe to keep legacy NULL-user rows shared, but account skills must
    # be allowed to use the same instruction content in their own scope.
    conn.execute("DROP INDEX IF EXISTS idx_skills_scope_content")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_scope_content "
        "ON skills (COALESCE(user_id, ''), COALESCE(project_id, ''), content_sha256)"
    )
    conn.commit()


def _add_auth_columns(conn: sqlite3.Connection) -> None:
    """Add auth identity, recovery, and MFA columns to old databases."""
    columns = (
        ("users", "organization_id", "TEXT REFERENCES organizations(id) ON DELETE SET NULL"),
        ("users", "recovery_key_hash", "TEXT"),
        ("users", "email", "TEXT"),
        ("users", "display_name", "TEXT"),
        ("users", "avatar_url", "TEXT"),
        ("users", "auth_provider", "TEXT"),
        ("users", "supabase_user_id", "TEXT"),
        ("auth_sessions", "mfa_verified", "INTEGER NOT NULL DEFAULT 1"),
    )
    for table, name, definition in columns:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            conn.commit()
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_supabase_user_id "
        "ON users (supabase_user_id) WHERE supabase_user_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_organization_id "
        "ON users (organization_id) WHERE organization_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_recovery_key_hash "
        "ON users (recovery_key_hash) WHERE recovery_key_hash IS NOT NULL"
    )
    conn.commit()


def _migrate_files_nullable_project(conn: sqlite3.Connection) -> None:
    """
    Make project_files.project_id nullable so files can be attached directly
    in quick chats (no project). Same probe-then-recreate pattern as the
    conversations migration.
    """
    try:
        conn.execute(
            "INSERT INTO project_files (id, project_id, filename, original_name, "
            "size_bytes, created_at) "
            "VALUES ('__migration_probe__', NULL, 'probe', 'probe', 0, 'x')"
        )
        conn.execute("DELETE FROM project_files WHERE id = '__migration_probe__'")
        conn.commit()
    except sqlite3.IntegrityError:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS project_files_v2 (
            id             TEXT PRIMARY KEY,
            project_id     TEXT REFERENCES projects(id) ON DELETE CASCADE,
            filename       TEXT NOT NULL,
            original_name  TEXT NOT NULL,
            mime_type      TEXT,
            size_bytes     INTEGER NOT NULL,
            extracted_text TEXT,
            created_at     TEXT NOT NULL,
            user_id        TEXT
        );
        INSERT OR IGNORE INTO project_files_v2
            SELECT id, project_id, filename, original_name, mime_type,
                   size_bytes, extracted_text, created_at, user_id
            FROM project_files;
        DROP TABLE project_files;
        ALTER TABLE project_files_v2 RENAME TO project_files;
        """)
        conn.commit()


def _add_run_columns(conn: sqlite3.Connection) -> None:
    """
    Add new columns to the runs table for databases created before these
    columns existed. Idempotent: silently skips columns that already exist.
    """
    new_columns = [
        ("routing_source",        "TEXT"),
        ("routing_reason",        "TEXT"),
        ("sticky_route_used",     "INTEGER DEFAULT 0"),
        ("cost_quality_tradeoff", "INTEGER"),
        ("fallback_chain",        "TEXT"),   # JSON array of model slugs
    ]
    for col_name, col_def in new_columns:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists — safe to ignore


def _add_runtime_settings_columns(conn: sqlite3.Connection) -> None:
    """Add runtime settings introduced after the initial table creation."""
    new_columns = [
        ("model_origin_policy", "TEXT NOT NULL DEFAULT 'any'"),
        ("allow_follow_up_questions", "INTEGER NOT NULL DEFAULT 1"),
    ]
    for col_name, col_def in new_columns:
        try:
            conn.execute(f"ALTER TABLE runtime_settings ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
