-- =============================================================================
-- SANADA Complete Supabase Setup & Trigger Fix
-- =============================================================================
-- Paste and run this complete query in your Supabase SQL Editor.
-- =============================================================================

-- 1. Drop the old broken trigger and function referencing 'profiles'
drop trigger if exists on_auth_user_created on auth.users;
drop function if exists public.handle_new_user() cascade;

-- 2. Add email column to public.app_users (safe if column already exists)
alter table public.app_users
add column if not exists email text unique;

-- 3. Procedure to register verified user into public.app_users
create or replace function public.register_verified_user (
  p_username text,
  p_password text,
  p_email text default null
) returns json language plpgsql security definer as $$
declare
  new_id uuid;
  clean_user text := trim(p_username);
  clean_mail text := lower(trim(coalesce(p_email, '')));
begin
  if exists (select 1 from public.app_users where lower(trim(username)) = lower(clean_user)) then
    return json_build_object('success', false, 'error', 'Username is already taken.');
  end if;

  insert into public.app_users (username, password, role, email)
  values (clean_user, p_password, 'user', nullif(clean_mail, ''))
  returning id into new_id;

  return json_build_object('success', true, 'user_id', new_id, 'username', clean_user, 'role', 'user'); 
exception
  when unique_violation then 
    return json_build_object('success', false, 'error', 'An account with this username or email already exists.');
  when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- 4. Allow login with EITHER username OR email
create or replace function public.authenticate_user (
  p_username text,
  p_password text
) returns table (user_id uuid, username text, role text) language plpgsql security definer as $$ 
begin
  return query
  select u.id, u.username, u.role
  from public.app_users u
  where (
    lower(trim(u.username)) = lower(trim(p_username))
    or (u.email is not null and lower(trim(u.email)) = lower(trim(p_username)))
  )
  and u.password = crypt(p_password, u.password);
end;
$$;

-- 5. Safe failsafe trigger to sync newly created auth.users into app_users
create or replace function public.handle_new_auth_user()
returns trigger as $$
begin
  insert into public.app_users (id, username, password, role, email)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'username', split_part(new.email, '@', 1)),
    'managed_by_supabase_auth',
    coalesce(new.raw_user_meta_data->>'role', 'user'),
    new.email
  )
  on conflict (username) do nothing;
  return new;
exception
  when others then
    -- Failsafe: never block auth.users signup
    return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_auth_user();

-- 6. Grant permissions
grant execute on function public.register_verified_user (text, text, text) to anon, authenticated;
grant execute on function public.authenticate_user (text, text) to anon, authenticated;
