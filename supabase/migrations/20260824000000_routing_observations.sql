-- Normalized Router V2 predictions, counterfactuals, and realized outcomes.
-- Raw prompts are never stored here; prompt_hash is a normalized SHA-256 digest.

create table if not exists public.routing_observations (
  id uuid primary key,
  decision_id text not null,
  run_id text,
  principal_id text,
  prompt_hash text not null,
  prompt_embedding jsonb,
  router_version text not null,
  lane text not null check (lane in ('general', 'code')),
  model text not null,
  role text not null check (role in ('candidate', 'selected', 'challenger')),
  propensity double precision,
  p_success double precision check (p_success between 0 and 1),
  p_success_lower double precision check (p_success_lower between 0 and 1),
  expected_output_tokens double precision,
  expected_cost_usd double precision,
  expected_latency_ms double precision,
  uncertainty double precision,
  eligible boolean not null default true,
  exclusion_reason text,
  realized_success double precision,
  realized_score double precision,
  realized_cost_usd double precision,
  realized_latency_ms double precision,
  input_tokens integer,
  output_tokens integer,
  evaluator text,
  evaluator_version text,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists routing_observations_decision_idx
  on public.routing_observations (decision_id, model);
create index if not exists routing_observations_router_created_idx
  on public.routing_observations (router_version, created_at desc);
create index if not exists routing_observations_principal_created_idx
  on public.routing_observations (principal_id, created_at desc);

alter table public.routing_observations enable row level security;
revoke all on public.routing_observations from anon, authenticated;

drop policy if exists "Router observations are readable by their owner"
  on public.routing_observations;
create policy "Router observations are readable by their owner"
  on public.routing_observations for select
  to authenticated
  using (principal_id = 'supabase:' || (select auth.uid())::text);

-- The FastAPI service role owns writes; browser clients never insert labels or
-- counterfactuals directly.
grant select, insert, update on public.routing_observations to service_role;
