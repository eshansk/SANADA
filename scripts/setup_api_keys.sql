-- =============================================================================
-- SANADA Web API Keys Schema & Verification Procedure
-- =============================================================================
-- Run this script in your Supabase SQL Editor to enable persistent API key
-- management and partner authentication for SANADA Machine Listening.
-- =============================================================================

-- 1. Create table public.api_keys
create table if not exists public.api_keys (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references public.app_users(id) on delete cascade,
    name text not null default 'Production API Key',
    key_prefix text not null,
    key_hash text not null,
    masked_key text not null,
    encrypted_key text,
    created_at timestamptz not null default now()
);

-- 2. Indexes for fast authentication lookups and user queries
create index if not exists idx_api_keys_hash on public.api_keys(key_hash);
create index if not exists idx_api_keys_user_id on public.api_keys(user_id);

-- 3. Enable Row-Level Security (RLS)
alter table public.api_keys enable row level security;

-- 4. Policies: Allow authenticated and anon users to manage API keys
create policy "Allow select api_keys" on public.api_keys
    for select using (true);

create policy "Allow insert api_keys" on public.api_keys
    for insert with check (true);

create policy "Allow delete api_keys" on public.api_keys
    for delete using (true);

-- 5. Grant access permissions to anon and authenticated roles
grant all on table public.api_keys to anon, authenticated, service_role;

-- 6. Helper function to verify an API key hash in PostgreSQL
create or replace function public.verify_api_key(p_key_hash text)
returns table (
    is_valid boolean,
    key_name text,
    user_id uuid,
    created_at timestamptz
) language plpgsql security definer as $$
begin
    return query
    select 
        true as is_valid,
        k.name as key_name,
        k.user_id,
        k.created_at
    from public.api_keys k
    where k.key_hash = p_key_hash
    limit 1;
end;
$$;

grant execute on function public.verify_api_key(text) to anon, authenticated, service_role;
