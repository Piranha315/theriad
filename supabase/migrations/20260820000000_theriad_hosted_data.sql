-- Hosted Theriad data plane.
--
-- Supabase Auth owns identity. These tables own tenant resources and are
-- protected by organization membership RLS. Provider secrets are represented
-- by a secret-store reference; plaintext API keys must never enter Postgres
-- or a VITE_* bundle.

create table if not exists public.organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  kind text not null default 'personal' check (kind in ('personal', 'team')),
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.organization_memberships (
  organization_id uuid not null references public.organizations(id) on delete cascade,
  -- Supports Supabase Auth ids and explicitly upgraded local account ids.
  user_id text not null,
  role text not null check (role in ('owner', 'admin', 'member', 'auditor')),
  personal boolean not null default false,
  created_at timestamptz not null default timezone('utc', now()),
  primary key (organization_id, user_id)
);

create unique index if not exists one_personal_org_per_user
  on public.organization_memberships (user_id) where personal = true;

create table if not exists public.billing_customers (
  organization_id uuid primary key references public.organizations(id) on delete cascade,
  stripe_customer_id text not null unique,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.billing_subscriptions (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  stripe_subscription_id text not null unique,
  stripe_customer_id text,
  plan_id text not null,
  interval text,
  status text not null,
  current_period_end timestamptz,
  cancel_at_period_end boolean not null default false,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.organization_entitlements (
  organization_id uuid primary key references public.organizations(id) on delete cascade,
  plan_id text not null default 'free',
  status text not null default 'active',
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.billing_events (
  event_id text primary key,
  event_type text not null,
  payload_sha256 text not null,
  received_at timestamptz not null default timezone('utc', now()),
  processed_at timestamptz
);

create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  name text not null,
  description text,
  instructions text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.project_settings (
  project_id uuid primary key references public.projects(id) on delete cascade,
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  allowed_model_patterns jsonb not null default '[]'::jsonb,
  blocked_model_patterns jsonb not null default '[]'::jsonb,
  max_cost_per_call_usd numeric(18, 8),
  allow_premium boolean not null default true,
  local_only boolean not null default false,
  enable_openrouter_auto_fallback boolean not null default false,
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  project_id uuid references public.projects(id) on delete set null,
  title text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.project_files (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  project_id uuid references public.projects(id) on delete set null,
  storage_path text not null,
  filename text not null,
  original_name text not null,
  mime_type text,
  size_bytes bigint not null default 0,
  extracted_text text,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.runs (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  project_id uuid references public.projects(id) on delete set null,
  conversation_id uuid references public.conversations(id) on delete set null,
  message_id uuid references public.messages(id) on delete set null,
  model text,
  provider text,
  tier text,
  policy text,
  input_tokens integer,
  output_tokens integer,
  actual_cost_usd numeric(18, 8),
  routing_latency_ms numeric(18, 3),
  provider_latency_ms numeric(18, 3),
  total_latency_ms numeric(18, 3),
  fallback_used boolean not null default false,
  status text not null,
  error text,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.memory_entries (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  project_id uuid references public.projects(id) on delete cascade,
  model_id text,
  label text not null,
  content text not null,
  source text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.memory_profiles (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  model_id text not null,
  context_enabled boolean not null default true,
  learning_enabled boolean not null default false,
  max_memories integer not null default 100 check (max_memories between 10 and 500),
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  unique (organization_id, user_id, model_id)
);

create table if not exists public.provider_connections (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  provider text not null,
  secret_ref text not null,
  active boolean not null default false,
  masked_key text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  unique (organization_id, provider)
);

create table if not exists public.usage_events (
  id uuid primary key default gen_random_uuid(),
  idempotency_key text not null unique,
  organization_id uuid not null references public.organizations(id) on delete cascade,
  user_id text not null,
  run_id uuid references public.runs(id) on delete set null,
  provider text,
  model text,
  input_tokens integer,
  output_tokens integer,
  provider_cost_usd numeric(18, 8),
  routing_fee_usd numeric(18, 8),
  total_billed_usd numeric(18, 8),
  status text not null default 'pending',
  billing_period text not null,
  stripe_invoice_item_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  reported_at timestamptz
);

create table if not exists public.audit_events (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid references public.organizations(id) on delete cascade,
  user_id text,
  event_type text not null,
  resource_type text,
  resource_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists projects_org_updated on public.projects (organization_id, updated_at desc);
create index if not exists conversations_org_updated on public.conversations (organization_id, updated_at desc);
create index if not exists runs_org_created on public.runs (organization_id, created_at desc);
create index if not exists usage_events_org_period on public.usage_events (organization_id, billing_period, created_at);

create or replace function public.is_theriad_member(target_org uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.organization_memberships m
    where m.organization_id = target_org and m.user_id = (select auth.uid())::text
  );
$$;

create or replace function public.theriad_role(target_org uuid)
returns text
language sql
stable
security definer
set search_path = public
as $$
  select m.role
  from public.organization_memberships m
  where m.organization_id = target_org
    and m.user_id = (select auth.uid())::text
  limit 1;
$$;

create or replace function public.is_theriad_writer(target_org uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(public.theriad_role(target_org) in ('owner', 'admin', 'member'), false);
$$;

create or replace function public.is_theriad_admin(target_org uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(public.theriad_role(target_org) in ('owner', 'admin'), false);
$$;

alter table public.organizations enable row level security;
alter table public.organization_memberships enable row level security;
alter table public.projects enable row level security;
alter table public.project_settings enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.project_files enable row level security;
alter table public.runs enable row level security;
alter table public.memory_entries enable row level security;
alter table public.memory_profiles enable row level security;
alter table public.provider_connections enable row level security;
alter table public.usage_events enable row level security;
alter table public.audit_events enable row level security;
alter table public.billing_events enable row level security;
alter table public.billing_customers enable row level security;
alter table public.billing_subscriptions enable row level security;
alter table public.organization_entitlements enable row level security;

do $$
declare
  table_name text;
  org_expression text;
begin
  foreach table_name in array array[
    'organizations', 'organization_memberships', 'projects', 'project_settings', 'conversations',
    'messages', 'project_files', 'runs', 'memory_entries',
    'memory_profiles', 'provider_connections', 'usage_events', 'audit_events',
    'billing_customers', 'billing_subscriptions', 'organization_entitlements'
  ] loop
    org_expression := case
      when table_name = 'organizations' then 'id'
      else 'organization_id'
    end;
    execute format('drop policy if exists "Theriad members can read %1$s" on public.%1$s', table_name);
    execute format('create policy "Theriad members can read %1$s" on public.%1$s for select to authenticated using (public.is_theriad_member(%2$s))', table_name, org_expression);
    execute format('drop policy if exists "Theriad members can write %1$s" on public.%1$s', table_name);
    execute format('create policy "Theriad members can write %1$s" on public.%1$s for all to authenticated using (public.is_theriad_writer(%2$s)) with check (public.is_theriad_writer(%2$s))', table_name, org_expression);
  end loop;
end $$;

drop policy if exists "Theriad members can write organizations" on public.organizations;
create policy "Theriad admins can write organizations"
  on public.organizations for all to authenticated
  using (public.is_theriad_admin(id))
  with check (public.is_theriad_admin(id));

drop policy if exists "Theriad members can write organization_memberships" on public.organization_memberships;
create policy "Theriad admins can write organization_memberships"
  on public.organization_memberships for all to authenticated
  using (public.is_theriad_admin(organization_id))
  with check (public.is_theriad_admin(organization_id));

drop policy if exists "Theriad members can write audit_events" on public.audit_events;
create policy "Theriad members can append audit_events"
  on public.audit_events for insert to authenticated
  with check (public.is_theriad_writer(organization_id));

-- Files are tenant-scoped under: theriad/<organization_id>/<file_id>.
insert into storage.buckets (id, name, public)
values ('theriad-files', 'theriad-files', false)
on conflict (id) do nothing;

drop policy if exists "Theriad members can read files" on storage.objects;
create policy "Theriad members can read files"
  on storage.objects for select to authenticated
  using (
    bucket_id = 'theriad-files'
    and (storage.foldername(name))[1] = 'theriad'
    and public.is_theriad_member(((storage.foldername(name))[2])::uuid)
  );

drop policy if exists "Theriad members can manage files" on storage.objects;
create policy "Theriad members can manage files"
  on storage.objects for all to authenticated
  using (
    bucket_id = 'theriad-files'
    and (storage.foldername(name))[1] = 'theriad'
    and public.is_theriad_writer(((storage.foldername(name))[2])::uuid)
  )
  with check (
    bucket_id = 'theriad-files'
    and (storage.foldername(name))[1] = 'theriad'
    and public.is_theriad_writer(((storage.foldername(name))[2])::uuid)
  );
