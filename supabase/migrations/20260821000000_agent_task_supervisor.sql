-- Shared durable state for TaskEngine, the canonical Theriad supervisor.
-- Apply after 20260801000000_theriad_profiles.sql.

create table if not exists public.agent_tasks (
  id uuid primary key,
  principal_id text,
  status text not null check (
    status in ('ready', 'running', 'waiting_for_approval', 'completed', 'failed', 'stopped')
  ),
  snapshot jsonb not null,
  revision bigint not null default 1 check (revision > 0),
  supervisor_id text,
  lease_expires_at timestamptz,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.agent_task_events (
  event_id uuid primary key,
  task_id uuid not null references public.agent_tasks(id) on delete cascade,
  principal_id text,
  sequence bigint not null check (sequence > 0),
  kind text not null,
  event jsonb not null,
  created_at timestamptz not null default timezone('utc', now()),
  unique (task_id, sequence)
);

create index if not exists agent_tasks_principal_updated_idx
  on public.agent_tasks (principal_id, updated_at desc);

create index if not exists agent_tasks_active_lease_idx
  on public.agent_tasks (status, lease_expires_at)
  where status in ('ready', 'running', 'waiting_for_approval');

create index if not exists agent_task_events_task_sequence_idx
  on public.agent_task_events (task_id, sequence);

alter table public.agent_tasks enable row level security;
alter table public.agent_task_events enable row level security;

-- FastAPI is the only runtime reader/writer. RLS remains defense-in-depth in
-- case a future migration grants authenticated reads; no browser role is
-- granted table access here.
revoke all on public.agent_tasks from anon, authenticated;
revoke all on public.agent_task_events from anon, authenticated;

drop policy if exists "Task owners can read their tasks" on public.agent_tasks;
create policy "Task owners can read their tasks"
  on public.agent_tasks for select
  to authenticated
  using (principal_id = 'supabase:' || (select auth.uid())::text);

drop policy if exists "Task owners can read their task events" on public.agent_task_events;
create policy "Task owners can read their task events"
  on public.agent_task_events for select
  to authenticated
  using (principal_id = 'supabase:' || (select auth.uid())::text);

create or replace function public.persist_agent_task(
  p_snapshot jsonb,
  p_event jsonb default null
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_task jsonb := p_snapshot -> 'task';
  v_task_id uuid := (p_snapshot #>> '{task,task_id}')::uuid;
  v_principal_id text := nullif(p_snapshot #>> '{task,principal_id}', '');
  v_status text := p_snapshot #>> '{task,status}';
  v_created_at timestamptz := to_timestamp((p_snapshot #>> '{task,created_at}')::double precision);
  v_updated_at timestamptz := to_timestamp((p_snapshot #>> '{task,updated_at}')::double precision);
begin
  if v_task is null or v_task_id is null or v_status is null then
    raise exception 'invalid agent task snapshot';
  end if;

  insert into public.agent_tasks (
    id, principal_id, status, snapshot, created_at, updated_at
  ) values (
    v_task_id, v_principal_id, v_status, p_snapshot, v_created_at, v_updated_at
  )
  on conflict (id) do update set
    principal_id = excluded.principal_id,
    status = excluded.status,
    snapshot = excluded.snapshot,
    revision = public.agent_tasks.revision + 1,
    updated_at = excluded.updated_at;

  if p_event is not null then
    insert into public.agent_task_events (
      event_id, task_id, principal_id, sequence, kind, event, created_at
    ) values (
      (p_event ->> 'event_id')::uuid,
      v_task_id,
      v_principal_id,
      (p_event ->> 'sequence')::bigint,
      p_event ->> 'kind',
      p_event,
      to_timestamp((p_event ->> 'created_at')::double precision)
    )
    on conflict (event_id) do nothing;
  end if;
end;
$$;

create or replace function public.claim_agent_task(
  p_task_id uuid,
  p_supervisor_id text,
  p_lease_seconds integer default 30
)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_updated integer;
begin
  update public.agent_tasks
  set supervisor_id = p_supervisor_id,
      lease_expires_at = timezone('utc', now()) + make_interval(secs => greatest(5, least(p_lease_seconds, 300))),
      updated_at = timezone('utc', now())
  where id = p_task_id
    and (
      supervisor_id = p_supervisor_id
      or lease_expires_at is null
      or lease_expires_at <= timezone('utc', now())
    );
  get diagnostics v_updated = row_count;
  return v_updated = 1;
end;
$$;

create or replace function public.heartbeat_agent_task(
  p_task_id uuid,
  p_supervisor_id text,
  p_lease_seconds integer default 30
)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_updated integer;
begin
  update public.agent_tasks
  set lease_expires_at = timezone('utc', now()) + make_interval(secs => greatest(5, least(p_lease_seconds, 300))),
      updated_at = timezone('utc', now())
  where id = p_task_id
    and supervisor_id = p_supervisor_id;
  get diagnostics v_updated = row_count;
  return v_updated = 1;
end;
$$;

create or replace function public.release_agent_task(
  p_task_id uuid,
  p_supervisor_id text
)
returns void
language sql
security definer
set search_path = ''
as $$
  update public.agent_tasks
  set supervisor_id = null,
      lease_expires_at = null,
      updated_at = timezone('utc', now())
  where id = p_task_id
    and supervisor_id = p_supervisor_id;
$$;

revoke all on function public.persist_agent_task(jsonb, jsonb) from public, anon, authenticated;
revoke all on function public.claim_agent_task(uuid, text, integer) from public, anon, authenticated;
revoke all on function public.heartbeat_agent_task(uuid, text, integer) from public, anon, authenticated;
revoke all on function public.release_agent_task(uuid, text) from public, anon, authenticated;

grant execute on function public.persist_agent_task(jsonb, jsonb) to service_role;
grant execute on function public.claim_agent_task(uuid, text, integer) to service_role;
grant execute on function public.heartbeat_agent_task(uuid, text, integer) to service_role;
grant execute on function public.release_agent_task(uuid, text) to service_role;
