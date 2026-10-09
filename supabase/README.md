# Theriad Supabase authentication

This integration uses Supabase Auth for Google identity and Supabase Postgres
for the user profile record. After Google sign-in, the FastAPI backend sends a
short-lived email code through the configured SMTP relay and only creates a
Theriad session after that code is verified. This is an email verification
step, not phishing-resistant MFA; an authenticator or security key is stronger.

## Configure the project

1. Create a Supabase project and run these migrations in order in the SQL editor:
   - `migrations/20260801000000_theriad_profiles.sql`
   - `migrations/20260820000000_theriad_hosted_data.sql`
   - `migrations/20260821000000_agent_task_supervisor.sql`
   - `migrations/20260824000000_routing_observations.sql`
   The second migration creates the organization-scoped hosted data plane,
   usage ledger, audit records, and the private `theriad-files` Storage bucket.
   The third creates durable task snapshots, append-only task events, and
   expiring supervisor leases.
2. In Supabase Auth → Providers → Google, enable Google and enter the Google
   OAuth client ID and secret. Google’s authorized redirect URI is the callback
   URL shown on that provider page, usually
   `https://<project-ref>.supabase.co/auth/v1/callback`.
3. Add the app origins to Supabase Auth → URL Configuration. For local work,
   add `http://localhost:5173/**` and `http://127.0.0.1:5173/**`.
4. Configure the SMTP relay in the server environment using the
   `THERIAD_SMTP_*` variables from the repository `.env.example`. Theriad uses
   this relay for the Google sign-in code and local-account recovery emails.

## Environment

Repository-root `.env` (used by both FastAPI and Vite):

```dotenv
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
SUPABASE_SECRET_KEY=sb_secret_...
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
THERIAD_APP_URL=https://<your-theriad-host>
THERIAD_SMTP_HOST=smtp.example.com
THERIAD_SMTP_PORT=587
THERIAD_SMTP_USERNAME=...
THERIAD_SMTP_PASSWORD=...
THERIAD_SMTP_FROM=Theriad <accounts@example.com>
THERIAD_SMTP_USE_TLS=true
THERIAD_TASK_STORE=supabase
THERIAD_ROUTING_OBSERVATION_STORE=auto
```

The publishable key is intended for browser use. Never put a Supabase
service-role key in a `VITE_*` variable or commit any secret.

`THERIAD_TASK_STORE=auto` selects Supabase only when both the URL and a
server-only secret are present; otherwise it uses local snapshots. Explicit
`supabase` mode fails closed instead of silently forking task history locally.
The adapter also accepts the legacy `SUPABASE_SERVICE_ROLE_KEY` name.

To preview an additive copy of existing local task snapshots, run
`python scripts/migrate_agent_tasks_to_supabase.py`. Add `--apply` only after
the migration and server-only key are configured. The copy is idempotent and
does not delete the local snapshots.

The shared task tables are inaccessible to browser roles and protected by
owner-scoped RLS as defense-in-depth. FastAPI is the sole runtime reader/writer
through a server-only key and restricted RPC functions. Expiring database leases
ensure only one TaskEngine supervisor advances a task at a time.

## Copy existing local tasks

Preview the additive migration first:

```bash
python scripts/migrate_agent_tasks_to_supabase.py
```

After the SQL migration and server-only key are configured:

```bash
python scripts/migrate_agent_tasks_to_supabase.py --apply
```

The copy is idempotent and leaves `data/agent-tasks` untouched. Only after the
copy succeeds should the backend be restarted with
`THERIAD_TASK_STORE=supabase`.

## Verify a user was persisted

After Google sign-in and email-code verification, run this in the Supabase SQL editor:

```sql
select id, email, display_name, provider, created_at, updated_at
from public.profiles
order by created_at desc;

select id, principal_id, status, revision, supervisor_id, lease_expires_at, updated_at
from public.agent_tasks
order by updated_at desc;

select router_version, lane, model, role, p_success, realized_success, created_at
from public.routing_observations
order by created_at desc
limit 100;
```

The app also upserts the signed-in profile through the authenticated Supabase
client and displays the Google identity plus “Email verified” in the Theriad
account area. The local Theriad session is marked as email-verified after the
code is consumed.

## Hosted data boundary

When `THERIAD_DEPLOYMENT_MODE=hosted`, Supabase is the intended system of record
for authenticated resources. SQLite remains the local/development adapter and
must not be used as a silent hosted fallback. The migration deliberately does
not claim anonymous legacy rows: import only records with an explicit source
account and organization mapping, then compare row counts and ownership before
cutting traffic over.

Provider API keys are represented by a secret-store reference in the hosted
schema. Keep the encryption key and the referenced provider values in the
deployment secret manager; never put them in Postgres plaintext, a Storage
object, a `VITE_*` variable, or a committed `.env` file.
