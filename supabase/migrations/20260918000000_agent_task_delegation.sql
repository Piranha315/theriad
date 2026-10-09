-- Add the durable parent wait state used by local model delegation.
-- Parent/child graph details remain in the versioned task snapshot so local
-- JSON and Supabase use the same canonical serialization contract.

alter table public.agent_tasks
  drop constraint if exists agent_tasks_status_check;

alter table public.agent_tasks
  add constraint agent_tasks_status_check check (
    status in (
      'ready',
      'running',
      'waiting_for_children',
      'waiting_for_approval',
      'completed',
      'failed',
      'stopped'
    )
  );

drop index if exists public.agent_tasks_active_lease_idx;

create index agent_tasks_active_lease_idx
  on public.agent_tasks (status, lease_expires_at)
  where status in (
    'ready',
    'running',
    'waiting_for_children',
    'waiting_for_approval'
  );
