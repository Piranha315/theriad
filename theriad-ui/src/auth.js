/**
 * auth.js — client-side session management for per-user data isolation.
 *
 * Visitors can start with a "ghost" free-plan session, and registering upgrades
 * that session in place so its history carries over.
 *
 * A fetch interceptor attaches the bearer token to every same-origin /api/
 * request so the many per-page apiFetch helpers (and raw FormData uploads)
 * are all authenticated without touching each call site.
 */

import { createClient } from '@supabase/supabase-js';

const CACHE_VERSION = 'v1';
const TOKEN_KEY = `theriad_auth_token:${CACHE_VERSION}`;
const USER_KEY = `theriad_auth_user:${CACHE_VERSION}`;
const LEGACY_TOKEN_KEY = 'theriad_auth_token';
const LEGACY_USER_KEY = 'theriad_auth_user';

// Only cache the identity fields needed to paint the shell immediately. The
// server remains the source of truth for permissions, credits, and profile
// details after the background refresh completes.
const CACHED_USER_FIELDS = [
  'id', 'username', 'display_name', 'email', 'is_guest',
  'auth_provider', 'avatar_url', 'mfa_verified', 'organization_id', 'organization_role',
];

let ensureSessionPromise = null;

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY
  || import.meta.env.VITE_SUPABASE_ANON_KEY;

// The publishable/anon key is safe to ship in the browser. Never put a
// Supabase service-role key in VITE_* variables or any other frontend file.
const supabase = SUPABASE_URL && SUPABASE_KEY
  ? createClient(SUPABASE_URL, SUPABASE_KEY, {
      auth: {
        flowType: 'pkce',
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    })
  : null;

export function isSupabaseConfigured() {
  return Boolean(supabase);
}

function readCachedValue(key, legacyKey) {
  try {
    const current = localStorage.getItem(key);
    if (current) return current;

    // Migrate the pre-versioned cache once, so existing users do not lose the
    // optimistic shell during the cache schema upgrade.
    const legacy = localStorage.getItem(legacyKey);
    if (legacy) {
      localStorage.setItem(key, legacy);
      localStorage.removeItem(legacyKey);
    }
    return legacy;
  } catch {
    return null;
  }
}

function cachedUserShape(user) {
  if (!user || typeof user !== 'object') return null;
  return Object.fromEntries(
    CACHED_USER_FIELDS
      .filter(field => user[field] !== undefined)
      .map(field => [field, user[field]])
  );
}

function cacheUser(user) {
  const cached = cachedUserShape(user);
  if (!cached) return;
  try { localStorage.setItem(USER_KEY, JSON.stringify(cached)); } catch { /* private mode or quota */ }
}

function authError(message, details = {}) {
  const error = new Error(message);
  error.code = 'SUPABASE_EMAIL_MFA_REQUIRED';
  error.details = details;
  return error;
}

function supabaseError(error, fallback) {
  return error instanceof Error ? error : new Error(error?.message || fallback);
}

function supabaseAuthError(error, fallback) {
  const result = supabaseError(error, fallback);
  if (!result.code) result.code = 'SUPABASE_AUTH_ERROR';
  return result;
}

export function getToken() {
  return readCachedValue(TOKEN_KEY, LEGACY_TOKEN_KEY);
}

export function getCachedUser() {
  try {
    const raw = readCachedValue(USER_KEY, LEGACY_USER_KEY);
    return cachedUserShape(raw ? JSON.parse(raw) : null);
  } catch {
    return null;
  }
}

export function hasAuthenticatedSession() {
  const token = getToken();
  const user = getCachedUser();
  return Boolean(token && user && !user.is_guest);
}

function storeSession(token, user) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch { /* private mode: session won't persist across reloads */ }
  cacheUser(user);
  notifyUserChanged(user);
}

export function clearSession() {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    localStorage.removeItem(LEGACY_USER_KEY);
  } catch { /* ignore */ }
  notifyUserChanged(null);
}

function notifyUserChanged(user) {
  window.dispatchEvent(new CustomEvent('theriad:user-changed', { detail: user }));
}

// ── Fetch interceptor ────────────────────────────────────────────────────────

let installed = false;

export function installAuthFetch() {
  if (installed) return;
  installed = true;
  const rawFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input?.url ?? '';
    const isApi = url.startsWith('/api/') && !url.startsWith('/api/auth/');
    const token = getToken();
    if (isApi && token) {
      const headers = new Headers(init.headers || (typeof input !== 'string' ? input.headers : undefined));
      if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
      init = { ...init, headers };
    }
    const res = await rawFetch(input, init);
    return res;
  };
}

// ── Auth API calls (bypass the interceptor's /api/auth/ exclusion on purpose) ─

async function authPost(path, body, withToken = false) {
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (withToken && token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, {
    method: 'POST',
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data?.detail;
    const msg = typeof detail === 'string'
      ? detail
      : detail?.message || data?.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

export async function fetchMe() {
  const token = getToken();
  if (!token) return null;
  const res = await fetch('/api/auth/me', {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) { clearSession(); return null; }
  if (!res.ok) return getCachedUser();
  const data = await res.json();
  cacheUser(data.user);
  notifyUserChanged(data.user);
  return data.user;
}

/** Restore a session, optionally creating a ghost account for guest mode. */
export function ensureSession({ createGuest = true } = {}) {
  if (ensureSessionPromise) return ensureSessionPromise;

  const cached = getCachedUser();
  ensureSessionPromise = (async () => {
    const existing = await fetchMe();

    // A verified Theriad account session is the app's authoritative login
    // state. Supabase also persists its Google session, but checking it first
    // would send a new email challenge on every reload. Keep the Supabase
    // branch below for guests, who may be returning from a Google sign-in and
    // still need to be upgraded into their registered account.
    if (existing && !existing.is_guest) return existing;

    if (supabase) {
      const { data: sessionData, error } = await supabase.auth.getSession();
      if (error) throw supabaseAuthError(error, 'Unable to restore the Google session.');
      if (sessionData.session) return completeSupabaseSession(sessionData.session);
    }

    if (existing) return existing;

    if (!createGuest) return null;

    const data = await authPost('/api/auth/guest');
    storeSession(data.token, data.user);
    return data.user;
  })()
    .catch(error => {
      // A cached identity is more useful than a blank screen when the network
      // is temporarily unavailable. Keep revalidation fail-soft; the next
      // focus or route load will try again.
      const supabaseAuthAttempt = [
        'SUPABASE_EMAIL_MFA_REQUIRED',
        'SUPABASE_AUTH_ERROR',
      ].includes(error?.code);
      if (cached && !supabaseAuthAttempt) {
        notifyUserChanged(cached);
        return cached;
      }
      throw error;
    })
    .finally(() => {
      ensureSessionPromise = null;
    });

  return ensureSessionPromise;
}

export async function register(username, password, email) {
  const data = await authPost('/api/auth/register', { username, password, email }, true);
  storeSession(data.token, data.user);
  return data.user;
}

export async function login(username, password) {
  const data = await authPost('/api/auth/login', { username, password });
  storeSession(data.token, data.user);
  return data.user;
}

export async function requestPasswordRecovery(identifier) {
  return authPost('/api/auth/recovery/request', { identifier });
}

export async function completePasswordRecovery(recoveryKey, password) {
  const data = await authPost('/api/auth/recovery/complete', {
    recovery_key: recoveryKey,
    password,
  });
  storeSession(data.token, data.user);
  return data.user;
}

export async function signInWithGoogle() {
  if (!supabase) {
    throw new Error('Google sign-in is not configured yet. Add the Supabase frontend variables and try again.');
  }
  const redirectTo = `${window.location.origin}${window.location.pathname}#/app`;
  const { error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: { redirectTo },
  });
  if (error) throw supabaseError(error, 'Unable to start Google sign-in.');
}

async function beginSupabaseEmailMfa(accessToken) {
  if (!supabase) throw new Error('Supabase authentication is not configured.');
  const data = await authPost('/api/auth/supabase/email/request', {
    access_token: accessToken,
  });
  return {
    stage: 'email',
    challengeId: data.challenge_id,
    email: data.email,
    expiresIn: data.expires_in,
  };
}

async function syncSupabaseProfile(user) {
  const metadata = user.user_metadata || {};
  const profile = {
    id: user.id,
    email: user.email || null,
    display_name: metadata.full_name || metadata.name || metadata.user_name || user.email || 'Theriad user',
    avatar_url: metadata.avatar_url || metadata.picture || null,
    provider: 'google',
  };
  const { data, error } = await supabase
    .from('profiles')
    .upsert(profile, { onConflict: 'id' })
    .select()
    .single();
  if (error) {
    throw new Error(`Supabase profile database is not ready: ${error.message}`);
  }
  return data || profile;
}

async function completeSupabaseSession(session) {
  try {
    const mfa = await beginSupabaseEmailMfa(session.access_token);
    throw authError('Check your email to finish signing in.', mfa);
  } catch (error) {
    if (error?.code === 'SUPABASE_EMAIL_MFA_REQUIRED') throw error;
    throw supabaseAuthError(error, 'Unable to send the Google verification email.');
  }
}

export async function resendSupabaseEmailCode() {
  if (!supabase) throw new Error('Supabase authentication is not configured.');
  const { data: sessionData, error } = await supabase.auth.getSession();
  if (error || !sessionData.session) throw supabaseError(error, 'The verified session could not be restored.');
  return beginSupabaseEmailMfa(sessionData.session.access_token);
}

export async function verifySupabaseEmailCode(challengeId, code) {
  if (!supabase) throw new Error('Supabase authentication is not configured.');
  const { data: sessionData, error } = await supabase.auth.getSession();
  if (error || !sessionData.session) throw supabaseError(error, 'The Google session could not be restored.');
  const supabaseUser = sessionData.session.user;
  const profile = await syncSupabaseProfile(supabaseUser);
  const data = await authPost('/api/auth/supabase/email/verify', {
    access_token: sessionData.session.access_token,
    challenge_id: challengeId,
    code,
  }, true);
  const user = {
    ...data.user,
    email: profile.email || supabaseUser.email,
    display_name: profile.display_name,
    avatar_url: profile.avatar_url,
    auth_provider: 'google',
    supabase_user_id: supabaseUser.id,
    mfa_verified: true,
  };
  storeSession(data.token, user);
  return user;
}

export async function cancelSupabaseSignIn() {
  if (supabase) await supabase.auth.signOut();
  clearSession();
}

export async function logout() {
  try { await authPost('/api/auth/logout', null, true); } catch { /* best-effort */ }
  if (supabase) {
    try { await supabase.auth.signOut(); } catch { /* best-effort */ }
  }
  clearSession();
}
