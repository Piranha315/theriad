import React, { useState, useEffect, useCallback, useTransition } from 'react';
import { ProjectsScreen, ProjectDetail } from './Projects.jsx';
import ChatPage from './Chat.jsx';
import RecentsPage from './Recents.jsx';
import MemoryScreen from './Memory.jsx';
import SkillsScreen from './Skills.jsx';
import ProviderSettings from './ProviderSettings.jsx';
import CostConfirmationDialog from './CostConfirmationDialog.jsx';
import SiteAnalyticsScreen from './SiteAnalytics.jsx';
import AppErrorBoundary from './components/AppErrorBoundary.jsx';
import { LoadingScreen } from './components/LoadingScreen.jsx';
import { DelayedLoader } from './components/LoadingState.jsx';
import { useDelayedLoading } from './components/useDelayedLoading.js';
import { toFriendlyError } from './uiErrors.js';
import { buildPreviewDoc, extractPreviewArtifact } from './previewArtifact.js';
import {
  cancelSupabaseSignIn, ensureSession, getCachedUser,
  completePasswordRecovery, isSupabaseConfigured, login as authLogin,
  logout as authLogout, requestPasswordRecovery,
  register as authRegister, resendSupabaseEmailCode,
  signInWithGoogle, verifySupabaseEmailCode,
} from './auth.js';

// --- API helper ---
async function apiFetch(path, options = {}) {
  // Destructure headers first so the spread of `rest` cannot overwrite them.
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    ...rest,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    const method = (rest.method ?? 'GET').toUpperCase();
    console.error('[Theriad] API error', { method, path, status: res.status, body });
    throw new Error(`API ${res.status} on ${method} ${path}: ${body}`);
  }
  return res.json();
}

const API_DOWN_MSG =
  'Theriad is having trouble reaching the workspace. Check your connection and try again.';

function isNetworkError(msg) {
  return (
    msg.includes('fetch') ||
    msg.includes('Failed') ||
    msg.includes('NetworkError') ||
    msg.includes('ECONNREFUSED') ||
    msg.includes('Load failed')
  );
}

// Returns a human-readable error string, replacing vague network errors with
// the actionable backend-down message.
function friendlyError(msg) {
  return isNetworkError(msg) ? API_DOWN_MSG : toFriendlyError(msg);
}

function formatLogTimestamp(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

// --- Icon Components (Inline SVG for zero dependencies) ---
const Icon = ({ path, className = 'w-5 h-5' }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>{path}</svg>
);

const GoogleMark = () => (
  <span className="theriad-auth-google-mark" aria-hidden="true">
    <svg viewBox="0 0 24 24" focusable="false">
      <path fill="#4285F4" d="M21.35 12.27c0-.79-.07-1.55-.2-2.27H12v4.3h5.24a4.48 4.48 0 0 1-1.94 2.94v2.45h3.14c1.84-1.69 2.91-4.18 2.91-7.42z" />
      <path fill="#34A853" d="M12 21.75c2.64 0 4.86-.87 6.48-2.36l-3.14-2.45c-.87.58-1.98.92-3.34.92-2.56 0-4.73-1.73-5.51-4.05H3.25v2.53A9.78 9.78 0 0 0 12 21.75z" />
      <path fill="#FBBC05" d="M6.49 13.81c-.2-.58-.31-1.2-.31-1.84s.11-1.26.31-1.84V7.6H3.25A9.75 9.75 0 0 0 2.25 12c0 1.57.38 3.06 1 4.37l3.24-2.56v0z" />
      <path fill="#EA4335" d="M12 5.13c1.44 0 2.73.5 3.75 1.48l2.81-2.81C16.86 2.22 14.64 1.25 12 1.25A9.78 9.78 0 0 0 3.25 7.6l3.24 2.53C7.27 6.86 9.44 5.13 12 5.13z" />
    </svg>
  </span>
);

const TheriadLogo = ({ className = 'w-28 h-36', imageClassName = 'w-full h-full object-contain' }) => (
  <div className={`${className} relative`}>
    <img src="/theriad-logo-transparent.png" alt="Theriad"
      className={imageClassName} />
  </div>
);

const Icons = {
  Analyze: (
    <Icon path={<><circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/></>} />
  ),
  Models: (
    <Icon path={<><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 12h.01M10 12h.01M14 12h.01M18 12h.01"/></>} />
  ),
  Costs: (
    <Icon path={<><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></>} />
  ),
  Logs: (
    <Icon path={<><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></>} />
  ),
  Settings: (
    <Icon path={<><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></>} />
  ),
  Search: (
    <Icon path={<><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></>} />
  ),
  Globe: (
    <Icon path={<><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18"/><path d="M12 3a14 14 0 0 0 0 18"/></>} />
  ),
  Copy: (
    <Icon path={<><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></>} />
  ),
  Alert: (
    <Icon path={<><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></>} />
  ),
  Check: (
    <Icon path={<><polyline points="20 6 9 17 4 12"/></>} />
  ),
};

// --- UI Primitives ---
const Badge = ({ children, color = 'cyan' }) => {
  const colors = {
    cyan:    'bg-accent/10 text-accent border-accent/30',
    emerald: 'bg-ok/10 text-ok border-ok/30',
    amber:   'bg-warn/10 text-warn border-warn/30',
    slate:   'bg-ink-faint/10 text-ink-muted border-line-strong',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${colors[color] ?? colors.slate}`}>
      {children}
    </span>
  );
};

const Spinner = ({ label = 'Loading…' }) => (
  <div className="flex items-center space-x-2 text-ink-muted text-sm py-4">
    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
    </svg>
    <span>{label}</span>
  </div>
);

const ApiError = ({ msg }) => {
  const networkError = isNetworkError(msg) || /^API 5\d\d/.test(msg);
  const message = networkError
    ? 'Theriad is having trouble reaching the workspace. Check your connection and try again.'
    : toFriendlyError(msg);
  return (
    <div className="rounded-xl border border-warn/25 bg-surface-raised p-5">
      <p className="text-sm font-medium text-ink">
        {networkError ? 'Can’t reach the backend' : 'Something went wrong'}
      </p>
      <p className="mt-1 text-sm text-ink-muted">
        {message}
      </p>
      {import.meta.env.DEV && (
        <details className="mt-3 text-xs text-ink-faint">
          <summary className="cursor-pointer hover:text-ink">Developer details</summary>
          <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-surface-sunken p-3 font-mono whitespace-pre-wrap">{msg}</pre>
        </details>
      )}
    </div>
  );
};

const EmptyState = ({ message }) => (
  <div className="text-center py-12 text-ink-faint text-sm">{message}</div>
);

function badgeColor(b) {
  if (b === 'Free' || b === 'Cheap') return 'emerald';
  if (b === 'Premium') return 'amber';
  return 'slate';
}

// ── Account section (sidebar footer): identity, auth actions, home link ───────
const AUTH_BACKDROP_USER = {
  display_name: 'Your workspace',
  username: 'Your workspace',
  is_guest: false,
};

const AccountSection = ({ preview = false }) => {
  const [user, setUser] = useState(() => (preview ? AUTH_BACKDROP_USER : getCachedUser()));
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const onChange = e => setUser(e.detail);
    window.addEventListener('theriad:user-changed', onChange);
    return () => window.removeEventListener('theriad:user-changed', onChange);
  }, []);

  async function handleLogout() {
    await authLogout();
    window.location.hash = '';
  }

  function openAuth(initialTab) {
    window.dispatchEvent(new CustomEvent('theriad:open-auth', {
      detail: { initialTab },
    }));
  }

  const isGuest = Boolean(user?.is_guest);
  const name = isGuest
    ? 'Not signed in'
    : (user?.display_name || user?.username || user?.email || 'Your workspace');
  const initials = isGuest ? 'G' : name.charAt(0).toUpperCase();

  return (
    <div className="theriad-account-section flex-shrink-0">
      <button
        type="button"
        onClick={() => setExpanded(value => !value)}
        aria-expanded={expanded}
        aria-label={isGuest ? 'Sign in' : 'Open account menu'}
        title={isGuest ? 'Sign in' : 'Account'}
        className={`theriad-account-trigger w-full flex items-center gap-3 text-left ${isGuest ? 'is-guest' : ''}`}
      >
        <span className="theriad-account-avatar" aria-hidden="true">{initials}</span>
        <span className="theriad-account-compact-label" aria-hidden="true">
          {isGuest ? 'Sign in' : 'Account'}
        </span>
        <span className="min-w-0 flex-1">
          <span className="theriad-account-label">Account</span>
          <span className="theriad-account-subline truncate">
            {isGuest ? 'Not signed in' : name}
          </span>
        </span>
        <Icon path={<polyline points="9 18 15 12 9 6" />} className="theriad-account-chevron" />
      </button>

      {expanded && (
        <div className="theriad-account-menu" role="group" aria-label="Account actions">
          {isGuest ? (
            <>
              <button type="button" onClick={() => openAuth('login')} className="theriad-account-menu__primary">
                Sign in
              </button>
              <button type="button" onClick={() => openAuth('register')} className="theriad-account-menu__secondary">
                Create account
              </button>
            </>
          ) : (
            <button type="button" onClick={handleLogout} className="theriad-account-menu__secondary theriad-account-menu__danger">
              Log out
            </button>
          )}
          <a href="#/" className="theriad-account-menu__secondary">
            ← Back to home page
          </a>
        </div>
      )}
    </div>
  );
};

// --- Sidebar ---
// ── Shared browser-backend status ─────────────────────────────────────────────
// Tiny pub/sub so the Sidebar reflects the *effective* browser backend as
// reported by the backend (session status / start response), instead of a
// hardcoded "Browser: Mock" label. Runtime views can publish resolved
// observation/action backends when they have fresher session data.
const browserBackendStore = {
  state: null, // { observation_backend, action_backend } | null
  listeners: new Set(),
  set(next) {
    this.state = next;
    this.listeners.forEach(fn => fn(next));
  },
  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  },
};

// The sidebar is a slide-over drawer below `lg` and a permanent rail from `lg`
// up (see the `lg:translate-x-0` class below). Only the drawer form has a
// closed state, so anything keyed off "closed" has to know which form is on
// screen — hence the matching media query rather than a bare `isOpen` check.
const DESKTOP_NAV_QUERY = '(min-width: 1024px)';

function useDesktopNav() {
  const [isDesktop, setIsDesktop] = useState(
    () => window.matchMedia(DESKTOP_NAV_QUERY).matches
  );

  useEffect(() => {
    const query = window.matchMedia(DESKTOP_NAV_QUERY);
    const update = () => setIsDesktop(query.matches);
    update();
    query.addEventListener?.('change', update);
    return () => query.removeEventListener?.('change', update);
  }, []);

  return isDesktop;
}

const Sidebar = ({ current, setCurrent, onNewChat, isOpen, onClose, user, preview = false }) => {
  const [providerStatus, setProviderStatus] = useState(null);
  const [backendUp, setBackendUp]           = useState(null); // null=checking, true=up, false=down
  const [toolsStatus, setToolsStatus]       = useState(null); // MCPBrokerStatus
  const [browserBackend, setBrowserBackend] = useState(browserBackendStore.state);
  const [statusOpen, setStatusOpen]         = useState(false);
  const isDesktopNav = useDesktopNav();
  // A closed drawer is translated off-screen but still rendered, so without
  // `inert` its twelve controls stay in the tab order and keyboard focus
  // walks off the side of the screen. `inert` also drops them from the
  // accessibility tree, and unlike `visibility: hidden` it leaves the
  // slide transition intact.
  const drawerClosed = !isDesktopNav && !isOpen;

  useEffect(() => browserBackendStore.subscribe(setBrowserBackend), []);

  useEffect(() => {
    function onProvidersChanged(event) {
      if (event.detail) setProviderStatus(event.detail);
    }
    window.addEventListener('theriad:providers-changed', onProvidersChanged);
    return () => window.removeEventListener('theriad:providers-changed', onProvidersChanged);
  }, []);

  useEffect(() => {
    if (preview) return undefined;

    // Health check — fast, no auth required
    apiFetch('/api/health')
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false));

    apiFetch('/api/settings/providers')
      .then(setProviderStatus)
      .catch(() => setProviderStatus(null));

    apiFetch('/api/tools/status')
      .then(setToolsStatus)
      .catch(() => setToolsStatus(null));

    apiFetch('/api/agent/browser/capabilities')
      .then(data => setBrowserBackend({
        observation_backend: data.observation_backend ?? 'mock',
        action_backend: data.action_backend ?? 'mock',
      }))
      .catch(() => {});
  }, [preview]);

  const activeProvider = providerStatus?.active_provider ?? 'theriad';
  const activeKeyProvider = activeProvider === 'theriad' ? 'openrouter' : activeProvider;
  const assistantConfigured = providerStatus?.providers?.[activeKeyProvider]?.configured ?? false;
  const localDeployment = providerStatus?.deployment_mode === 'local'
    || providerStatus?.deployment_mode === 'development';
  const accountRequiredForProvider = (user?.is_guest || preview) && !localDeployment;
  const assistantLabel = activeProvider === 'theriad'
    ? 'Theriad Router'
    : activeProvider === 'openai'
    ? 'OpenAI'
    : 'Anthropic';

  // The navigation is deliberately grouped the way the workspace is used:
  // work first, then projects and insights, with the lower-frequency controls
  // still visible instead of hidden behind a generic "More" drawer.
  const navGroups = [
    {
      label: 'Work',
      items: [
        { id: 'Chat',    icon: <Icon path={<><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></>} /> },
        { id: 'Recents', icon: <Icon path={<><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></>} /> },
      ],
    },
    {
      label: 'Projects',
      items: [
        { id: 'Projects', icon: <Icon path={<><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></>} /> },
      ],
    },
    {
      label: 'Insights',
      items: [
        { id: 'Costs', icon: Icons.Costs },
        { id: 'Logs',  icon: Icons.Logs },
      ],
    },
    {
      label: 'Advanced',
      items: [
        { id: 'Models',   icon: Icons.Models },
        { id: 'Settings', icon: Icons.Settings },
        { id: 'Skills',   icon: <Icon path={<><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><path d="M9 7h6M9 11h4"/></>} /> },
        { id: 'Memory',   icon: <Icon path={<><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="16" r="2.5"/><circle cx="19" cy="16" r="2.5"/><path d="M10.9 7.2 6.3 13.9m6.8-6.7 4.6 6.7M7.5 16h9"/></>} /> },
      ],
    },
  ];

  function handleNavClick(id) {
    setCurrent(id);
    // Close drawer on mobile after navigation
    onClose?.();
  }

  function renderNavButton(item) {
    return (
      <button
        key={item.id}
        type="button"
        onClick={() => handleNavClick(item.id)}
        aria-current={current === item.id ? 'page' : undefined}
        aria-label={item.id}
        title={item.id}
        data-tooltip={item.id}
          className={`theriad-nav-button w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors duration-150 text-sm font-medium ${
            current === item.id
              ? 'bg-accent/10 text-accent'
              : 'text-ink-muted hover:bg-surface-hover hover:text-ink'
          }`}
      >
        {item.icon}
        <span>{item.id}</span>
      </button>
    );
  }

  return (
    <div
      inert={drawerClosed}
      className={`theriad-sidebar w-72 bg-surface border-r border-line flex flex-col fixed h-full z-50 transition-transform duration-200 ease-in-out lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}
    >
      <div className="theriad-sidebar-brand border-b border-line relative">
        <div className="theriad-sidebar-brand-lockup">
          <TheriadLogo className="theriad-sidebar-logo" />
          <div className="theriad-sidebar-brand-copy">
            <strong>Theriad</strong>
            <span>Cost-aware AI workspace</span>
          </div>
        </div>
      </div>
      <button
        type="button"
        className="theriad-new-chat-button"
        onClick={() => {
          window.dispatchEvent(new CustomEvent('theriad:new-chat'));
          onNewChat?.();
          onClose?.();
        }}
        aria-label="Start a new chat"
        title="New chat"
      >
        <Icon path={<><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z"/></>} />
        <span>New chat</span>
      </button>
      <nav className="theriad-sidebar-nav flex-1 min-h-0 p-3 space-y-1 overflow-y-auto" aria-label="Workspace navigation">
        {navGroups.map(group => (
          <section key={group.label} className="theriad-sidebar-group">
            <p className="theriad-sidebar-section-label">{group.label}</p>
            <div className="space-y-1">
              {group.items.map(renderNavButton)}
            </div>
          </section>
        ))}
      </nav>
      {!(localDeployment && assistantConfigured) && (
        <div className="theriad-plan-card">
          <div className="theriad-plan-card__accent" aria-hidden="true" />
          <p className="theriad-plan-card__title">
            {accountRequiredForProvider ? 'Free guest plan' : localDeployment ? 'Local workspace' : 'Personal workspace'}
          </p>
          <p className="theriad-plan-card__copy">
            {accountRequiredForProvider
              ? 'Connect a provider for paid calls.'
              : 'Add an OpenRouter key to enable local calls.'}
          </p>
          <button
            type="button"
            className="theriad-plan-card__action"
            aria-label={accountRequiredForProvider ? 'Sign in' : 'Manage providers'}
            title={accountRequiredForProvider ? 'Sign in' : 'Manage providers'}
            onClick={() => {
              if (accountRequiredForProvider) {
                window.dispatchEvent(new CustomEvent('theriad:open-auth', { detail: { initialTab: 'login' } }));
              } else {
                handleNavClick('Settings');
              }
            }}
          >
            <span className="theriad-plan-card__action-label-full">
              {accountRequiredForProvider ? 'Sign in' : 'Manage providers'}
            </span>
            <span className="theriad-plan-card__action-label-short">
              {accountRequiredForProvider ? 'Sign in' : 'Manage'}
            </span>
          </button>
        </div>
      )}
      <AccountSection preview={preview} />
      <div className="theriad-status-panel p-3 border-t border-line text-xs text-ink-faint flex-shrink-0">
        <button
          type="button"
          onClick={() => setStatusOpen(value => !value)}
          aria-expanded={statusOpen}
          className="w-full flex items-center justify-between gap-3 rounded-lg px-2 py-2 text-left hover:bg-surface-hover transition-colors"
        >
          <span className="flex items-center gap-2 min-w-0">
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${backendUp === false ? 'bg-danger' : !assistantConfigured ? 'bg-warn' : backendUp === true ? 'bg-ok' : 'bg-line-strong'}`} />
            <span className="truncate text-ink-muted">System status</span>
          </span>
          <span aria-hidden="true">{statusOpen ? '−' : '+'}</span>
        </button>
        {statusOpen && <div className="px-2 pb-2 pt-1">
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
            backendUp === null ? 'bg-line-strong' : backendUp ? 'bg-ok' : 'bg-danger'
          }`}></div>
          <span className="truncate">Backend: {backendUp === null ? 'Checking…' : backendUp ? 'Connected' : 'Unavailable'}</span>
        </div>
        <div className="flex items-center space-x-2 mt-2">
          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${assistantConfigured ? 'bg-ok' : 'bg-warn'}`}></div>
          <span className="truncate">Assistant: {assistantLabel} / {assistantConfigured ? 'Connected' : 'No key'}</span>
        </div>
        <div className="flex items-center space-x-2 mt-2">
          <div className="w-2 h-2 rounded-full bg-line-strong flex-shrink-0"></div>
          <span>Ollama: Local</span>
        </div>
        {(() => {
          const cdp = toolsStatus?.servers?.find(s => s.server_name === 'chrome_devtools');
          const dot = cdp == null
            ? 'bg-line-strong'
            : cdp.available ? 'bg-ok' : 'bg-warn';
          const label = cdp == null
            ? 'Checking…'
            : cdp.available ? 'Connected' : 'Not configured';
          return (
            <div className="flex items-center space-x-2 mt-2">
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`}></div>
              <span className="truncate">Chrome MCP: {label}</span>
            </div>
          );
        })()}
        {(() => {
          // Effective browser backend from capabilities or fresher session
          // state. Falls back to Mock before either is available.
          const obsB = browserBackend?.observation_backend ?? 'mock';
          const actB = browserBackend?.action_backend ?? 'mock';
          const parts = [];
          if (obsB === 'playwright') parts.push('Live');
          if (actB === 'playwright') parts.push('Real');
          const label = parts.length ? `Playwright / ${parts.join(' / ')}` : 'Mock';
          const dot = parts.length ? 'bg-ok' : 'bg-warn';
          return (
            <div className="flex items-center space-x-2 mt-2">
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`}></div>
              <span className="truncate">Browser: {label}</span>
            </div>
          );
        })()}
        </div>}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Workflow Plan Panel (v2)
// ─────────────────────────────────────────────────────────────────────────────
const WORKFLOW_TIER_COLOR = {
  no_model:         'text-ink-faint',
  code_agent:       'text-info',
  cheap_bulk:       'text-ok',
  cheap_code:       'text-ok',
  code_strong:      'text-accent',
  strong_reasoning: 'text-accent',
  premium:          'text-warn',
  free_or_floor:    'text-ok',
};

const ROUTE_LATENCY_ESTIMATES = {
  no_model: 500,
  cheap_bulk: 2000,
  cheap_code: 8000,
  code_strong: 8000,
  strong_reasoning: 5000,
  premium: 9000,
};

function estimatedRouteLatency(tier) {
  return ROUTE_LATENCY_ESTIMATES[tier] ?? 2000;
}

const WorkflowPanel = ({ data, loading, error }) => {
  if (loading) return <div className="mb-6"><Spinner label="Classifying workflow…" /></div>;
  if (error)   return <div className="mb-6"><ApiError msg={error} /></div>;
  if (!data)   return null;

  if (!data.workflow_selected) {
    return (
      <div className="mb-6 bg-surface-raised border border-line rounded-xl px-5 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2 text-sm text-ink-muted">
          <span className="w-1.5 h-1.5 rounded-full bg-line-strong inline-block"></span>
          <span>No workflow detected — standard v1 routing applies.</span>
        </div>
        <div className="text-xs text-ink-faint text-right">
          <span>{data.decision_source} · conf {data.confidence?.toFixed(2)}</span>
          <span className="block text-ink-muted">Standard route quality fit: {data.confidence >= 0.8 ? 'high' : 'moderate'}</span>
        </div>
      </div>
    );
  }

  const stepCosts = Object.fromEntries((data.step_costs ?? []).map(step => [step.name, step]));
  const qualityFit = data.confidence >= 0.85 ? 'High' : data.confidence >= 0.7 ? 'Moderate' : 'Guarded';

  return (
    <div className="mb-6 bg-surface-raised border border-info/30 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-6 py-3 border-b border-line bg-info/5 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-2 h-2 rounded-full bg-info"></div>
          <span className="text-sm font-semibold text-info">Workflow Plan</span>
          <Badge color="cyan">{data.workflow_type}</Badge>
          <span className="text-xs text-ink-faint">{data.decision_source} · conf {data.confidence?.toFixed(2)}</span>
        </div>
          <span className="text-xs text-ink-faint">{data.steps.length} steps · {qualityFit} quality fit</span>
      </div>

      {/* Cost summary */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4 px-6 py-4 border-b border-line bg-surface">
        <div>
          <p className="text-xs text-ink-faint mb-1">Theriad Est. Cost</p>
          <p className="font-mono text-lg font-semibold text-ink">${data.estimated_total_cost_usd.toFixed(5)}</p>
        </div>
        <div>
          <p className="text-xs text-ink-faint mb-1">Baseline ({data.baseline})</p>
          <p className="font-mono text-lg font-semibold text-ink-faint line-through">${data.baseline_estimated_cost_usd.toFixed(5)}</p>
        </div>
        <div>
          <p className="text-xs text-ink-faint mb-1">Est. Savings</p>
          <p className="font-mono text-lg font-semibold text-ok">{data.estimated_savings_pct}%</p>
        </div>
        <div>
          <p className="text-xs text-ink-faint mb-1">Est. Latency</p>
          <p className="font-mono text-lg font-semibold text-ink">{data.estimated_latency_ms.toLocaleString()}ms</p>
        </div>
        <div>
          <p className="text-xs text-ink-faint mb-1">Confidence / quality</p>
          <p className="font-mono text-lg font-semibold text-ok">{(data.confidence * 100).toFixed(0)}% · {qualityFit}</p>
        </div>
      </div>

      {/* Step list */}
      <div className="p-5 space-y-2">
        {data.steps.map((step, i) => (
          <div key={step.name} className="flex items-start gap-3 bg-surface rounded-lg px-4 py-3 border border-line">
            <span className="text-xs font-mono text-ink-faint mt-0.5 w-3 flex-shrink-0">{i + 1}</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-sm font-mono text-ink">{step.name}</span>
                {!step.uses_model && <Badge color="slate">no model</Badge>}
              </div>
              <p className="text-xs text-ink-faint truncate">{step.description}</p>
              <p className="text-[10px] text-ink-muted mt-1">
                Selected: <span className="font-mono text-accent">{stepCosts[step.name]?.model ?? (step.uses_model ? 'resolved by tier' : 'local tool')}</span>
                {' · '}{stepCosts[step.name]?.estimated_latency_ms ?? 0}ms
                {' · '}${Number(stepCosts[step.name]?.estimated_cost_usd ?? 0).toFixed(5)}
              </p>
            </div>
            <span className={`text-xs font-mono flex-shrink-0 ${WORKFLOW_TIER_COLOR[step.routing_tier] ?? 'text-ink-muted'}`}>
              {step.routing_tier}
            </span>
          </div>
        ))}
        {data.reason && (
          <p className="text-xs text-ink-faint pt-1 px-1">{data.reason}</p>
        )}
      </div>
    </div>
  );
};

// ChromeQAPanel — Chrome MCP status indicator + QA results (below iframe)
// ─────────────────────────────────────────────────────────────────────────────
function ChromeQAPanel({ artifact, tailwindEnabled }) {
  const [chromeSt, setChromeSt]   = useState(null);
  const [qaResult, setQaResult]   = useState(null);
  const [qaRunning, setQaRunning] = useState(false);
  const [qaError, setQaError]     = useState(null);

  // Load Chrome status once on mount
  useEffect(() => {
    apiFetch('/api/runtime/status')
      .then(setChromeSt)
      .catch(() => setChromeSt(null));
  }, []);

  const chromeAvailable = chromeSt?.chrome_mcp_status === 'connected';

  const handleQA = async () => {
    if (!artifact) return;
    setQaRunning(true);
    setQaResult(null);
    setQaError(null);
    try {
      const res = await apiFetch('/api/preview/qa', {
        method: 'POST',
        body: JSON.stringify({
          html:             artifact.html,
          css:              artifact.css ?? '',
          tailwind_enabled: tailwindEnabled,
        }),
      });
      setQaResult(res);
    } catch (e) {
      setQaError(friendlyError(e.message));
    } finally {
      setQaRunning(false);
    }
  };

  // Status indicator line
  const statusDot   = chromeSt == null
    ? 'bg-line-strong'
    : chromeAvailable ? 'bg-ok' : 'bg-warn';
  const statusLabel = chromeSt == null
    ? 'Checking Chrome MCP…'
    : chromeAvailable
    ? 'Chrome MCP: Connected — QA in Chrome available'
    : 'Chrome MCP: Not configured — iframe preview still works';

  return (
    <div className="mt-3 border-t border-line pt-3 space-y-3">
      {/* Status row */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-ink-faint">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot}`}></span>
          <span>{statusLabel}</span>
        </div>
        {chromeAvailable && artifact && (
          <button
            onClick={handleQA}
            disabled={qaRunning}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-accent/10 text-accent border border-accent/30 hover:bg-accent/20 transition-colors disabled:opacity-50"
          >
            {qaRunning ? 'Running QA…' : 'QA in Chrome'}
          </button>
        )}
      </div>

      {/* QA error */}
      {qaError && (
        <div className="text-xs text-danger bg-danger/10 border border-danger/30 rounded-lg px-3 py-2">
          {qaError}
        </div>
      )}

      {/* QA running */}
      {qaRunning && (
        <div className="text-xs text-ink-faint animate-pulse">
          Opening preview in Chrome, capturing screenshot and diagnostics…
        </div>
      )}

      {/* QA result */}
      {qaResult && (
        <div className="space-y-2 text-xs">
          {/* Status */}
          <div className={`flex items-center gap-2 font-medium ${
            qaResult.status === 'ok' ? 'text-ok' : qaResult.status === 'unavailable' ? 'text-warn' : 'text-danger'
          }`}>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
              qaResult.status === 'ok' ? 'bg-ok' : qaResult.status === 'unavailable' ? 'bg-warn' : 'bg-danger'
            }`}></span>
            QA {qaResult.status} · {qaResult.latency_ms}ms
          </div>

          {qaResult.message && qaResult.status !== 'ok' && (
            <p className="text-ink-faint">{qaResult.message}</p>
          )}

          {/* Screenshot */}
          {qaResult.screenshot && (
            <div>
              <p className="text-ink-muted mb-1">Screenshot</p>
              <img
                src={`data:image/png;base64,${qaResult.screenshot}`}
                alt="Chrome QA screenshot"
                className="rounded border border-line max-w-full"
                style={{ maxHeight: '300px', objectFit: 'contain' }}
              />
            </div>
          )}

          {/* DOM summary */}
          {qaResult.dom_summary && (
            <div>
              <p className="text-ink-muted mb-0.5">DOM</p>
              <p className="font-mono text-ink">{qaResult.dom_summary}</p>
            </div>
          )}

          {/* Console errors */}
          {qaResult.console_errors?.length > 0 && (
            <div>
              <p className="text-danger mb-1">Console errors ({qaResult.console_errors.length})</p>
              <ul className="space-y-0.5">
                {qaResult.console_errors.map((e, i) => (
                  <li key={i} className="font-mono text-danger/80 truncate">{e}</li>
                ))}
              </ul>
            </div>
          )}
          {qaResult.console_errors?.length === 0 && qaResult.status === 'ok' && (
            <p className="text-ok">No console errors.</p>
          )}

          {/* Network errors */}
          {qaResult.network_errors?.length > 0 && (
            <div>
              <p className="text-warn mb-1">Network errors ({qaResult.network_errors.length})</p>
              <ul className="space-y-0.5">
                {qaResult.network_errors.map((e, i) => (
                  <li key={i} className="font-mono text-warn/80 truncate">{e}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {qaResult.warnings?.length > 0 && (
            <div>
              <p className="text-ink-faint mb-1">Warnings</p>
              {qaResult.warnings.map((w, i) => (
                <p key={i} className="text-ink-faint">⚠ {w}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// PreviewPanel — sandboxed iframe renderer for HTML/CSS artifacts
const PreviewPanel = ({ artifact, tailwindEnabled, onToggleTailwind }) => {
  if (!artifact) {
    return (
      <div className="p-6 min-w-0">
        <div className="flex items-center justify-center h-32 text-ink-faint text-sm border border-dashed border-line rounded-lg">
          No renderable HTML detected in this output.
        </div>
      </div>
    );
  }

  const srcDoc  = buildPreviewDoc(artifact, tailwindEnabled);
  // allow-scripts is required when Tailwind CDN is on; never combined with allow-same-origin
  const sandbox = 'allow-scripts';
  const previewKey = `${artifact.kind}_${artifact.source}_${tailwindEnabled}_${artifact.html.length}`;

  return (
    <div className="p-4 min-w-0">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <div className="flex items-center gap-3 text-xs text-ink-faint flex-wrap">
          <span className="font-mono bg-surface-hover px-2 py-0.5 rounded">{artifact.kind}</span>
          <span className="text-ink-faint">via {artifact.source}</span>
          {artifact.warnings.map((w, i) => (
            <span key={i} className="text-warn">⚠ {w}</span>
          ))}
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-1.5 text-xs text-ink-muted cursor-pointer select-none">
            <input
              type="checkbox"
              checked={tailwindEnabled}
              onChange={e => onToggleTailwind(e.target.checked)}
              className="accent-accent"
            />
            Tailwind CDN
          </label>
        </div>
      </div>

      {/* Sandbox notice */}
      <p className="text-xs text-ink-faint mb-2">
        🔒 Preview is sandboxed — scripts run in an isolated origin with no network access{tailwindEnabled ? '; Tailwind CDN allowed' : ''}.
      </p>

      {/* iframe */}
      <div
        className="w-full overflow-hidden rounded-lg border border-line bg-white"
        style={{ maxWidth: '100%' }}
      >
        <iframe
          key={previewKey}
          srcDoc={srcDoc}
          sandbox={sandbox}
          title="Preview"
          style={{ width: '100%', height: '600px', border: 'none', display: 'block', maxWidth: '100%' }}
        />
      </div>

      {/* Chrome QA panel (optional — does not affect iframe) */}
      <ChromeQAPanel artifact={artifact} tailwindEnabled={tailwindEnabled} />
    </div>
  );
};

// ExecResultCard — tabbed Output / Preview / Diagnostics view for execution results
const ExecResultCard = ({
  execResult, activeTab, setActiveTab,
  tailwindEnabled, setTailwindEnabled,
  copied, onCopy,
  showAttempts, setShowAttempts,
}) => {
  const hasOutput = !!(execResult.output && execResult.output.trim());
  const artifact  = extractPreviewArtifact(execResult.output ?? '');

  return (
    <div className={`mb-6 bg-surface-raised rounded-xl overflow-hidden border ${
      !hasOutput ? 'border-warn/30' : 'border-ok/30'
    }`}>
      {/* Header + Tabs */}
      <div className={`flex items-center justify-between px-6 py-3 border-b border-line ${
        !hasOutput ? 'bg-warn/5' : 'bg-ok/5'
      }`}>
        <div className="flex items-center gap-3 flex-wrap">
          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${!hasOutput ? 'bg-warn' : 'bg-ok'}`}></div>
          <span className={`text-sm font-semibold ${!hasOutput ? 'text-warn' : 'text-ok'}`}>
            Model Output
          </span>
          {execResult.fallback_used && <Badge color="amber">Fallback Used</Badge>}
          {/* Tabs */}
          <div className="flex items-center gap-1 ml-2 border-l border-line pl-3">
            {['output', 'preview', 'diagnostics'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-2.5 py-1 text-xs rounded capitalize transition-colors ${
                  activeTab === tab
                    ? 'bg-surface-hover text-ink'
                    : 'text-ink-faint hover:text-ink'
                }`}
              >
                {tab}
                {tab === 'preview' && artifact && (
                  <span className="ml-1 inline-block w-1.5 h-1.5 rounded-full bg-accent align-middle"></span>
                )}
              </button>
            ))}
          </div>
        </div>
        {activeTab === 'output' && hasOutput && (
          <button
            onClick={onCopy}
            className="flex items-center space-x-1.5 text-xs text-ink-muted hover:text-ink bg-surface-hover hover:bg-surface-hover px-3 py-1.5 rounded-lg transition-colors flex-shrink-0"
          >
            <span>{Icons.Copy}</span>
            <span>{copied ? 'Copied!' : 'Copy Output'}</span>
          </button>
        )}
      </div>

      {/* Output Tab */}
      {activeTab === 'output' && (
        <div className="p-6 min-w-0">
          {!hasOutput ? (
            <div className="bg-warn/10 border border-warn/30 rounded-lg p-4 flex items-start space-x-3">
              <span className="text-warn flex-shrink-0 mt-0.5">{Icons.Alert}</span>
              <div>
                <p className="text-sm font-medium text-warn mb-1">No renderable output returned</p>
                <p className="text-xs text-warn/80">
                  The provider reported{' '}
                  <span className="font-mono">{execResult.output_tokens ?? 0}</span> output tokens
                  but the extracted content was empty. See Diagnostics tab for
                  <span className="font-mono"> finish_reason</span> and fallback details.
                </p>
              </div>
            </div>
          ) : (
            <pre className="text-sm text-ink whitespace-pre-wrap break-words font-sans leading-relaxed max-w-full overflow-x-auto">
              {execResult.output}
            </pre>
          )}
        </div>
      )}

      {/* Preview Tab */}
      {activeTab === 'preview' && (
        <PreviewPanel
          artifact={artifact}
          tailwindEnabled={tailwindEnabled}
          onToggleTailwind={setTailwindEnabled}
        />
      )}

      {/* Diagnostics Tab */}
      {activeTab === 'diagnostics' && (
        <div className="px-6 py-4 bg-surface">
          {/* Summary row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-xs mb-3">
            <div>
              <p className="text-ink-faint mb-1">Model Used</p>
              <p className="font-mono text-accent truncate">{execResult.selected_model}</p>
            </div>
            <div>
              <p className="text-ink-faint mb-1">Tier</p>
              <p className="font-mono text-ink">{execResult.selected_tier}</p>
            </div>
            <div>
              <p className="text-ink-faint mb-1">Tokens (in / out)</p>
              <p className="font-mono text-ink">
                {execResult.input_tokens} / {execResult.output_tokens}
              </p>
            </div>
            <div>
              <p className="text-ink-faint mb-1">Actual Cost</p>
              <p className="font-mono text-ok">
                {execResult.actual_cost_usd != null
                  ? `$${execResult.actual_cost_usd.toFixed(5)}`
                  : <span className="text-ink-faint italic">pricing unavailable</span>}
              </p>
            </div>
            <div>
              <p className="text-ink-faint mb-1">Total Latency</p>
              <p className="font-mono text-ink">
                {(execResult.total_latency_ms ?? execResult.latency_ms ?? 0).toFixed(0)}ms
              </p>
            </div>
            <div>
              <p className="text-ink-faint mb-1">Provider Latency</p>
              <p className="font-mono text-ink">
                {(execResult.provider_latency_ms ?? 0).toFixed(0)}ms
              </p>
            </div>
          </div>

          {/* Latency breakdown */}
          {execResult.routing_latency_ms != null && (
            <div className="flex items-center gap-3 text-xs text-ink-faint mb-3 pb-3 border-b border-line">
              <span className="text-ink-faint">Latency breakdown:</span>
              <span>routing <span className="font-mono text-ink-muted">{execResult.routing_latency_ms.toFixed(0)}ms</span></span>
              <span className="text-ink-faint">·</span>
              <span>provider <span className="font-mono text-ink-muted">{execResult.provider_latency_ms.toFixed(0)}ms</span></span>
              {execResult.fallback_latency_ms > 0 && (
                <>
                  <span className="text-ink-faint">·</span>
                  <span>fallback <span className="font-mono text-warn">{execResult.fallback_latency_ms.toFixed(0)}ms</span></span>
                </>
              )}
              <span className="text-ink-faint">·</span>
              <span>total <span className="font-mono text-ink-muted">{execResult.total_latency_ms.toFixed(0)}ms</span></span>
            </div>
          )}

          {/* Execution attempts (collapsible) */}
          {execResult.execution_attempts && execResult.execution_attempts.length > 0 && (
            <div>
              <button
                onClick={() => setShowAttempts(a => !a)}
                className="flex items-center gap-2 text-xs text-ink-faint hover:text-ink transition-colors mb-2"
              >
                <span className={`transition-transform ${showAttempts ? 'rotate-90' : ''}`}>▶</span>
                <span>
                  Execution Attempts ({execResult.execution_attempts.length} called
                  {execResult.fallback_available?.length > 0
                    ? `, ${execResult.fallback_available.length} fallback available`
                    : ''})
                </span>
              </button>
              {showAttempts && (
                <div className="space-y-2">
                  {execResult.execution_attempts.map((att, i) => (
                    <div
                      key={i}
                      className={`rounded-lg px-4 py-3 border text-xs font-mono space-y-1 ${
                        att.status === 'success'
                          ? 'bg-ok/5 border-ok/20'
                          : 'bg-danger/5 border-danger/20'
                      }`}
                    >
                      <div className="flex items-center gap-3 mb-1">
                        <span className="text-ink-faint">Attempt {att.attempt}</span>
                        <span className={att.status === 'success' ? 'text-ok' : 'text-danger'}>
                          {att.status}
                        </span>
                        {att.timeout_triggered && (
                          <span className="text-warn">timeout</span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-0.5 text-ink-muted">
                        <span>model <span className="text-accent">{att.model}</span></span>
                        <span>tier <span className="text-ink">{att.tier}</span></span>
                        <span>latency <span className="text-ink">{att.latency_ms?.toFixed(0)}ms</span></span>
                        {att.finish_reason != null && (
                          <span>finish <span className={
                            att.finish_reason === 'stop' ? 'text-ok'
                            : att.finish_reason === 'length' ? 'text-warn'
                            : 'text-danger'
                          }>{att.finish_reason}</span></span>
                        )}
                        {att.tokens_per_second != null && (
                          <span>tok/s <span className="text-ink">{att.tokens_per_second}</span></span>
                        )}
                        {att.input_tokens != null && (
                          <span>in <span className="text-ink">{att.input_tokens}</span></span>
                        )}
                        {att.output_tokens != null && (
                          <span>out <span className="text-ink">{att.output_tokens}</span></span>
                        )}
                        {att.fallback_reason && (
                          <span className="col-span-2">reason <span className="text-warn">{att.fallback_reason}</span></span>
                        )}
                        {att.error && (
                          <span className="col-span-4 text-danger break-all">{att.error}</span>
                        )}
                      </div>
                    </div>
                  ))}
                  {execResult.fallback_available?.length > 0 && (
                    <div className="rounded-lg px-4 py-2 border border-line bg-surface text-xs text-ink-faint">
                      <span className="text-ink-faint">Fallback available (not needed): </span>
                      {execResult.fallback_available.map((m, i) => (
                        <React.Fragment key={i}>
                          {i > 0 && <span className="mx-1">→</span>}
                          <span className="font-mono text-ink-faint">{m}</span>
                        </React.Fragment>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Legacy chain — hidden when new diagnostics available */}
          {(!execResult.execution_attempts || execResult.execution_attempts.length === 0) && execResult.fallback_chain && execResult.fallback_chain.length > 0 && (
            <div className="flex items-center space-x-2 text-xs">
              <span className="text-ink-faint">Chain:</span>
              {execResult.fallback_chain.map((m, i) => (
                <React.Fragment key={i}>
                  {i > 0 && <span className="text-ink-faint">→</span>}
                  <span className={`font-mono ${i === 0 && !execResult.fallback_used ? 'text-ok' : 'text-ink-faint'}`}>{m}</span>
                </React.Fragment>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Analyze Screen — unified routing, workflow, cost, and execution view
// ─────────────────────────────────────────────────────────────────────────────
const AnalyzeScreen = () => {
  const [prompt, setPrompt]           = useState('Write a Python function that takes a list of URLs and returns their HTTP status codes concurrently using asyncio.');
  const policy = 'balanced';
  // Keep the wire fields at their existing defaults while routing stays automatic.
  const speedPreference = 50;
  const qualityPreference = 50;
  const [baseline, setBaseline]       = useState('openai/gpt-6-luna');
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [pendingCost, setPendingCost] = useState(null);
  // Execution state
  const [execResult, setExecResult]   = useState(null);
  const [execLoading, setExecLoading] = useState(false);
  const [execError, setExecError]     = useState(null);
  const [copied, setCopied]           = useState(false);
  const [showAttempts, setShowAttempts] = useState(false);
  // Preview tab state
  const [activeTab, setActiveTab]           = useState('output');
  const [tailwindEnabled, setTailwindEnabled] = useState(false);
  // Workflow analysis state (v2)
  const [wfResult, setWfResult]       = useState(null);
  const [wfError, setWfError]         = useState(null);

  const policyLabel = {
    balanced: 'Balanced',
    aggressive_savings: 'Aggressive Savings',
    quality_first: 'Quality First',
    free_moe_experimental: 'Free MoE (Experimental)',
  };

  async function handleAnalyze() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setWfError(null);
    // Clear previous execution output when re-analyzing
    setExecResult(null);
    setExecError(null);
    try {
      const [routeData, workflowData] = await Promise.all([
        apiFetch('/api/route', {
          method: 'POST',
          body: JSON.stringify({
            prompt, policy, baseline,
            speed_preference: speedPreference,
            quality_preference: qualityPreference,
          }),
        }),
        apiFetch('/api/workflow/analyze', {
          method: 'POST',
          body: JSON.stringify({
            prompt, policy, baseline,
            speed_preference: speedPreference,
            quality_preference: qualityPreference,
          }),
        }),
      ]);
      setResult(routeData);
      setWfResult(workflowData);
    } catch (e) {
      setError(e.message);
      setResult(null);
      setWfResult(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleExecuteConfirm(costDecision = null) {
    setExecLoading(true);
    setExecError(null);
    setExecResult(null);
    setActiveTab('output');
    try {
      if (!costDecision) {
        const preview = await apiFetch('/api/chat/cost-preview', {
          method: 'POST',
          body: JSON.stringify({
            message: prompt,
            policy,
            baseline,
            speed_preference: speedPreference,
            quality_preference: qualityPreference,
          }),
        });
        if (preview.requires_confirmation) {
          setPendingCost(preview);
          return;
        }
      }
      const data = await apiFetch('/api/execute', {
        method: 'POST',
        body: JSON.stringify({
          prompt, policy, baseline, allow_paid_calls: true,
          speed_preference: speedPreference,
          quality_preference: qualityPreference,
          workflow_selected: wfResult?.workflow_selected ?? false,
          cost_decision: costDecision,
        }),
      });
      setExecResult(data);
    } catch (e) {
      setExecError(e.message);
    } finally {
      setExecLoading(false);
    }
  }

  function handleCopyOutput() {
    if (!execResult?.output) return;
    navigator.clipboard?.writeText(execResult.output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  const estTokens = Math.max(1, Math.round(prompt.split(/\s+/).filter(Boolean).length * 1.33));

  return (
    <div className="max-w-6xl mx-auto pb-12">
      {/* Controls */}
      <div className="mb-6 max-w-md">
        <div>
          <label className="block text-xs font-medium text-ink-muted mb-1.5 uppercase tracking-wider">Baseline</label>
          <select
            value={baseline}
            onChange={e => setBaseline(e.target.value)}
            className="w-full bg-surface-raised border border-line rounded-lg py-2 pl-3 pr-8 text-sm text-ink outline-none focus:border-accent/50 font-mono truncate"
          >
            <option value="openai/gpt-6-luna">openai/gpt-6-luna</option>
            <option value="anthropic/claude-opus-5">anthropic/claude-opus-5</option>
            <option value="deepseek/deepseek-v4.1-flash">deepseek/deepseek-v4.1-flash</option>
          </select>
        </div>
      </div>

      {/* Prompt input */}
      <div className="bg-surface-raised border border-line rounded-xl mb-6 overflow-hidden">
        <textarea
          className="w-full bg-transparent p-5 text-ink outline-none resize-none placeholder-ink-faint text-sm font-sans min-h-[120px]"
          placeholder="Enter a prompt to analyze routing, workflow, cost, and quality…"
          value={prompt}
          onChange={e => { setPrompt(e.target.value); setResult(null); setError(null); setWfResult(null); setWfError(null); }}
        />
        <div className="flex flex-col gap-3 p-4 border-t border-line bg-surface sm:flex-row sm:justify-between sm:items-start">
          <div className="text-xs text-ink-faint space-y-0.5 min-w-0 flex-1">
            <div>
              <span className="text-ink-muted">Est. Tokens:</span>{' '}
              <span className="font-mono text-ink">~{estTokens}</span>
            </div>
            <div className="text-ink-faint italic leading-snug">
              Analysis is offline. Execute uses the analyzed route and your provider contract.
            </div>
          </div>
          <div className="flex flex-wrap justify-end gap-2 flex-shrink-0">
            <button
              onClick={handleAnalyze}
              disabled={loading || !prompt.trim()}
              className="bg-accent/20 hover:bg-accent/30 disabled:opacity-40 text-accent px-5 py-2 rounded-lg text-sm font-medium transition-colors border border-accent/30"
            >
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
            <button
              onClick={() => handleExecuteConfirm()}
              disabled={!result || loading || execLoading}
              className="bg-warn/15 hover:bg-warn/25 disabled:opacity-40 text-warn px-5 py-2 rounded-lg text-sm font-medium transition-colors border border-warn/30"
            >
              Execute
            </button>
          </div>
        </div>
      </div>

      {error && <div className="mb-6"><ApiError msg={error} /></div>}
      {loading && <div className="mb-6"><Spinner label="Routing prompt…" /></div>}

      <WorkflowPanel data={wfResult} loading={loading} error={wfError} />

      {/* Execution loading + output */}
      {execLoading && (
        <div className="mb-6 bg-surface-raised border border-line rounded-xl p-6">
          <Spinner label="Calling provider — real inference in progress…" />
        </div>
      )}

      {execError && !execLoading && (
        <div className="mb-6">
          <div className="bg-surface-raised border border-danger/30 rounded-xl p-6">
            <p className="text-xs text-danger uppercase tracking-wider mb-2 font-medium">Execution Failed</p>
            <ApiError msg={execError} />
          </div>
        </div>
      )}

      {execResult && !execLoading && (
        <ExecResultCard
          execResult={execResult}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          tailwindEnabled={tailwindEnabled}
          setTailwindEnabled={setTailwindEnabled}
          copied={copied}
          onCopy={handleCopyOutput}
          showAttempts={showAttempts}
          setShowAttempts={setShowAttempts}
        />
      )}


      {result && !loading ? (
        <div className="space-y-6">
          {/* Result card */}
          <div className="bg-surface-raised border border-line rounded-xl p-6">
            <div className="flex justify-between items-start mb-6">
              <div>
                <p className="text-xs text-ink-muted uppercase tracking-wider mb-1">Selected Tier & Model</p>
                <h2 className="text-2xl font-bold text-ink font-mono">{result.model}</h2>
                <p className="text-sm text-accent mt-1">
                  Tier: <span className="font-mono">{result.tier}</span>
                  {' '}via {result.provider}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-ink-muted uppercase tracking-wider mb-1">Confidence</p>
                <p className="text-2xl font-bold text-ok">{result.confidence.toFixed(2)}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4 border-t border-line pt-6">
              <div>
                <p className="text-xs text-ink-faint mb-1">Est. Cost</p>
                <p className="text-lg font-semibold text-ink font-mono">
                  ${result.estimated_cost_usd === 0 ? '0.00000' : result.estimated_cost_usd.toFixed(5)}
                </p>
              </div>
              <div>
                <p className="text-xs text-ink-faint mb-1">Baseline Cost</p>
                <p className="text-lg font-semibold text-ink-muted line-through font-mono">
                  ${result.baseline_cost_usd === 0 ? '0.00000' : result.baseline_cost_usd.toFixed(5)}
                </p>
              </div>
              <div>
                <p className="text-xs text-ink-faint mb-1">Est. Savings</p>
                <p className="text-lg font-semibold text-ok font-mono">{result.savings_pct}%</p>
              </div>
              <div>
                <p className="text-xs text-ink-faint mb-1">Est. Latency</p>
                <p className="text-lg font-semibold text-ink font-mono">~{estimatedRouteLatency(result.tier).toLocaleString()}ms</p>
              </div>
              <div>
                <p className="text-xs text-ink-faint mb-1">Tokens (In / Out)</p>
                <p className="text-lg font-semibold text-ink font-mono">
                  ~{result.estimated_prompt_tokens} / {result.estimated_output_tokens}
                </p>
              </div>
            </div>

            {result.fallback_chain && result.fallback_chain.length > 0 && (
              <div className="mt-6 bg-surface border border-line rounded-lg p-4">
                <p className="text-xs text-ink-faint mb-2 uppercase tracking-wider">Fallback Chain</p>
                <div className="flex flex-wrap items-center gap-2 font-mono text-sm">
                  <Badge color="emerald">Primary</Badge>
                  <span className="text-ink">{result.model}</span>
                  {result.fallback_chain.map(([prov, model], i) => (
                    <React.Fragment key={i}>
                      <span className="text-ink-faint">→</span>
                      <Badge color="slate">Fallback</Badge>
                      <span className="text-ink">{model}</span>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Explain panel */}
          <div className="bg-surface-raised border border-line rounded-xl p-6">
            <h3 className="text-lg font-semibold text-ink mb-4">Explain Decision</h3>
            <div className="space-y-4 text-sm">
              <div className="flex">
                <div className="w-36 text-ink-faint flex-shrink-0">Layer 0 Rules</div>
                <div className="text-ink">
                  {result.routing_path
                    .filter(r => r.layer.startsWith('L0:') && r.fired)
                    .map(r => (
                      <span key={r.layer} className="mr-3">
                        <span className="font-mono text-accent">{r.layer.replace('L0:', '')}</span>
                        {r.reason && r.reason !== 'abstained' ? ` — ${r.reason}` : ''}
                      </span>
                    ))}
                  {result.routing_path.filter(r => r.layer.startsWith('L0:') && r.fired).length === 0 && (
                    <span className="text-ink-faint">No L0 rules fired (all abstained)</span>
                  )}
                </div>
              </div>
              <div className="flex">
                <div className="w-36 text-ink-faint flex-shrink-0">Feature Signals</div>
                <div className="text-ink">
                  token_count={result.features.token_count},
                  {' '}code_ratio=<span className="font-mono text-ink">{result.features.code_ratio}</span>,
                  {' '}json_prob=<span className="font-mono text-ink">{result.features.json_probability}</span>
                </div>
              </div>
              <div className="flex">
                <div className="w-36 text-ink-faint flex-shrink-0">Winning Layer</div>
                <div className="text-ink">
                  <span className="font-mono text-ink">{result.winning_layer}</span>
                  {' '}→ tier <span className="font-mono text-accent">{result.tier}</span>
                  {' '}at confidence <span className="font-mono text-ok">{result.confidence}</span>
                </div>
              </div>
              <div className="flex">
                <div className="w-36 text-ink-faint flex-shrink-0">Policy Decision</div>
                <div className="text-ink">
                  <span className="font-mono text-ink">{policyLabel[policy] || policy}</span>
                  {' '}policy selected <span className="font-mono text-accent">{result.model}</span>
                  {result.savings_pct > 0 && ` (${result.savings_pct}% cheaper than baseline)`}
                </div>
              </div>
              <div className="flex bg-ok/5 border border-ok/20 rounded-lg p-3 mt-4">
                <div className="w-36 text-ok flex-shrink-0 font-medium">Summary</div>
                <div className="text-ok/80">
                  Theriad selected <span className="font-mono">{result.tier}</span> because {result.reason}.
                  Premium models were not used because expected quality risk is low.
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : !loading && !error ? (
        <div className="bg-surface-raised border border-dashed border-line rounded-xl p-12 text-center">
          <div className="w-12 h-12 mx-auto rounded-full bg-surface-hover flex items-center justify-center text-ink-faint mb-4">
            {Icons.Analyze}
          </div>
          <h3 className="text-lg font-medium text-ink">Analyze your next AI request</h3>
          <p className="text-sm text-ink-faint mt-2 max-w-md mx-auto">
            Paste a prompt to see its workflow, selected models, routing rationale, cost, latency, confidence, and baseline comparison.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto mt-8">
            {[
              { n: '01', label: 'Decompose the workflow' },
              { n: '02', label: 'Compare cost and latency' },
              { n: '03', label: 'Execute the analyzed request' },
            ].map(({ n, label }) => (
              <div key={n} className="bg-surface p-4 rounded-lg border border-line text-left">
                <p className="text-accent text-xs font-mono mb-1">{n}</p>
                <p className="text-ink text-sm font-medium">{label}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {pendingCost && (
        <CostConfirmationDialog
          preview={pendingCost}
          onPaid={() => { setPendingCost(null); handleExecuteConfirm('paid'); }}
          onFree={() => { setPendingCost(null); handleExecuteConfirm('free'); }}
          onCancel={() => setPendingCost(null)}
        />
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. Models Screen
// ─────────────────────────────────────────────────────────────────────────────
const TIER_LABELS = {
  free_or_floor: 'Free or floor',
  cheap_bulk: 'Cheap bulk',
  cheap_code: 'Everyday coding',
  code_strong: 'Strong coding',
  strong_reasoning: 'Strong reasoning',
  premium: 'Premium fallback',
  free_moe_general: 'Free general expert',
  free_moe_code: 'Free coding expert',
  free_moe_verify: 'Free verification expert',
  free_moe_summarize: 'Free summarization expert',
};

const ModelsScreen = () => {
  const [models, setModels]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [modelOriginPolicy, setModelOriginPolicy] = useState('any');
  const [originSaving, setOriginSaving] = useState(false);

  useEffect(() => {
    Promise.all([apiFetch('/api/models'), apiFetch('/api/settings/runtime')])
      .then(([modelData, settings]) => {
        setModels(modelData);
        setModelOriginPolicy(settings.model_origin_policy ?? 'any');
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function changeModelOriginPolicy(value) {
    const previous = modelOriginPolicy;
    setModelOriginPolicy(value);
    setOriginSaving(true);
    try {
      await apiFetch('/api/settings/runtime', {
        method: 'PUT',
        body: JSON.stringify({ model_origin_policy: value }),
      });
      const refreshed = await apiFetch('/api/models');
      setModels(refreshed);
    } catch (e) {
      setModelOriginPolicy(previous);
      setError(e.message);
    } finally {
      setOriginSaving(false);
    }
  }

  if (loading) return <DelayedLoader loading={loading} label="Loading tier registry…" />;
  if (error)   return <ApiError msg={error} />;

  return (
    <div className="pb-12">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-surface-raised p-4">
        <div>
          <p className="text-sm font-medium text-ink">Model origin</p>
          <p className="mt-1 text-xs text-ink-muted">A hard routing requirement, separate from tier risk.</p>
        </div>
        <select
          value={modelOriginPolicy}
          onChange={event => changeModelOriginPolicy(event.target.value)}
          disabled={originSaving}
          aria-label="Model origin policy"
          className="rounded-lg border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/60 disabled:opacity-60"
        >
          <option value="any">Any origin</option>
          <option value="us_only">US-only models</option>
        </select>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map(m => (
          <div key={m.tier} className="bg-surface-raised border border-line rounded-xl p-5 hover:border-accent/30 transition-colors">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-sm font-semibold text-ink">{TIER_LABELS[m.tier] ?? m.tier}</h3>
                <p className="text-[10px] font-mono text-accent mt-0.5">{m.tier}</p>
                <p className="text-xs text-ink-muted mt-2 leading-relaxed">{m.use}</p>
              </div>
            </div>
            <div className="mb-4 pb-4 border-b border-line">
              <p className="text-xs text-ink-faint mb-1">Primary Model</p>
              <p className="text-sm font-mono text-ink">{m.model}</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs mb-4">
                <div><p className="text-ink-faint mb-1">Provider</p><p className="text-ink">{m.provider}{m.origin ? ` · ${m.origin}` : ''}</p></div>
              <div><p className="text-ink-faint mb-1">Context</p><p className="text-ink font-mono">{m.ctx}</p></div>
              <div><p className="text-ink-faint mb-1">Price (In)</p><p className="text-ink font-mono">{m.p_in}/1M</p></div>
              <div><p className="text-ink-faint mb-1">Price (Out)</p><p className="text-ink font-mono">{m.p_out}/1M</p></div>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex space-x-1">
                {m.badges.map(b => <Badge key={b} color={badgeColor(b)}>{b}</Badge>)}
              </div>
              <div className="text-xs text-ink-faint">Risk: <span className="text-ink">{m.risk}</span></div>
            </div>
            {m.fallback_chain && m.fallback_chain.length > 0 && (
              <div className="mt-3 pt-3 border-t border-line">
                <p className="text-xs text-ink-faint mb-1">Fallbacks</p>
                {m.fallback_chain.map((f, i) => (
                  <p key={i} className="text-xs font-mono text-ink-faint">{f.model}</p>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. Costs Screen
// ─────────────────────────────────────────────────────────────────────────────
const CostsScreen = () => {
  const [stats, setStats]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    apiFetch('/api/stats')
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <DelayedLoader loading={loading} label="Loading cost data…" />;
  if (error)   return <ApiError msg={error} />;

  if (!stats || !stats.has_data) {
    return (
      <div className="pb-12">
        <EmptyState message="No inference records yet. Cost data will appear here after running prompts through Theriad." />
      </div>
    );
  }

  const requestCount = Number(stats.total_requests ?? 0);
  const totalCost = Number(stats.total_cost_usd ?? 0);
  const unknownCostRequests = Number(stats.unknown_cost_requests ?? 0);
  const costsComplete = stats.costs_complete !== false;
  const dailyCosts = (stats.daily_costs ?? []).map(day => ({
    ...day,
    cost: Number(day.cost ?? 0),
  }));
  const blendedRate = stats.blended_cost_per_mtok;
  const baselineRate = stats.baseline_blended_cost_per_mtok;
  const maxDailyCost = Math.max(...dailyCosts.map(d => d.cost), 0.01);
  const estimatedSavingsUsd = stats.estimated_savings_usd == null
    ? null : Number(stats.estimated_savings_usd);
  const estimatedSavingsPct = stats.estimated_savings_pct == null
    ? null : Number(stats.estimated_savings_pct);
  const costStatusMessage = stats.cost_status === 'recorded_zero'
    ? `Recorded spend is $0.00 across ${requestCount.toLocaleString()} request${requestCount === 1 ? '' : 's'}.`
    : !costsComplete
      ? `${unknownCostRequests.toLocaleString()} request${unknownCostRequests === 1 ? '' : 's'} do not have a recorded provider cost, so total spend and savings are incomplete.`
      : `Costs are recorded across ${requestCount.toLocaleString()} request${requestCount === 1 ? '' : 's'}.`;
  const allBars = dailyCosts.map(d => Math.max(4, Math.round(d.cost / maxDailyCost * 100)));
  const barData = dailyCosts.length < 14
    ? [...Array(14 - dailyCosts.length).fill(null), ...dailyCosts].map((d, i) => ({
        date: d?.date ?? '', cost: d?.cost ?? 0,
        height: d ? allBars[i - (14 - dailyCosts.length)] : 0,
      }))
    : dailyCosts.slice(-14).map((d, i) => ({
        date: d.date, cost: d.cost, height: allBars[allBars.length - 14 + i],
      }));

  return (
    <div className="pb-12">
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-4 mb-6">
        {[
          { label: 'Total Spend',      val: costsComplete ? `$${totalCost.toFixed(2)}` : '—', color: 'text-ink',
            sub: stats.cost_status === 'recorded_zero' ? 'Recorded zero-cost usage' : !costsComplete ? 'Some costs unavailable' : null },
          { label: 'Projected $/1M tok', val: blendedRate != null ? `$${blendedRate.toFixed(2)}` : '—',
            color: 'text-accent',
            sub: blendedRate != null && baselineRate != null ? `vs $${baselineRate.toFixed(2)} on ${stats.baseline_model}` : null },
          { label: 'Est. Savings',     val: estimatedSavingsUsd != null ? `$${estimatedSavingsUsd.toFixed(4)}` : '—', color: estimatedSavingsUsd != null && estimatedSavingsUsd >= 0 ? 'text-ok' : 'text-ink-muted' },
          { label: 'Requests Routed',  val: requestCount.toLocaleString(), color: 'text-ink' },
          { label: 'Avg Cost / Req',   val: costsComplete && requestCount > 0
              ? `$${(totalCost / requestCount).toFixed(4)}`
              : costsComplete ? '$0.00' : '—', color: 'text-accent' },
          { label: 'Fallback Rate',    val: `${stats.fallback_rate}%`,                  color: 'text-warn' },
          { label: 'Premium Usage',    val: `${stats.premium_usage_pct}%`,              color: 'text-warn' },
        ].map(stat => (
          <div key={stat.label} className="bg-surface-raised border border-line rounded-xl p-4">
            <p className="text-xs text-ink-faint uppercase tracking-wider mb-1.5">{stat.label}</p>
            <p className={`text-xl font-bold font-mono ${stat.color}`}>{stat.val}</p>
            {stat.sub && <p className="mt-1 text-[10px] text-ink-faint leading-snug">{stat.sub}</p>}
          </div>
        ))}
      </div>

      <p className="-mt-3 mb-6 text-xs text-ink-faint" role="status">
        {costStatusMessage}{' '}
        {costsComplete && estimatedSavingsPct != null
          ? `Savings compare Theriad’s routed cost with the same token usage on ${stats.baseline_model}; ${estimatedSavingsPct.toFixed(1)}% saved.`
          : !costsComplete
            ? 'Savings will be shown once every recorded request has a known cost.'
            : ''}
      </p>

      <div className="bg-surface-raised border border-line rounded-xl p-6 mb-6">
        <h3 className="text-sm font-medium text-ink mb-4">Daily Cost Trend (Last 14 Days)</h3>
        <div className="h-56 grid grid-cols-[repeat(14,minmax(0,1fr))] gap-1.5 items-end">
          {barData.map((bar, i) => (
            <div key={bar.date || `empty-${i}`} className="min-w-0 h-full flex flex-col justify-end" title={bar.date ? `${bar.date}: $${bar.cost.toFixed(6)}` : 'No data'}>
              <div
                className="bg-accent/50 hover:bg-accent-strong transition-colors rounded-sm"
                style={{ height: `${bar.height}%` }}
              ></div>
              <span className="mt-1 text-[9px] text-ink-faint text-center truncate">{bar.date ? bar.date.slice(5) : '—'}</span>
            </div>
          ))}
        </div>
      </div>

      {stats.recent_high_cost && stats.recent_high_cost.length > 0 && (
        <div className="bg-surface-raised border border-line rounded-xl overflow-hidden">
          <div className="p-4 border-b border-line">
            <h3 className="text-sm font-medium text-ink">Recent High-Cost Requests</h3>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-surface text-ink-faint text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left p-4 font-medium">Reason / Preview</th>
                <th className="text-left p-4 font-medium">Tier</th>
                <th className="text-left p-4 font-medium">Model</th>
                <th className="text-right p-4 font-medium">Cost</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {stats.recent_high_cost.map((row, i) => (
                <tr key={i} className="hover:bg-surface-hover">
                  <td className="p-4 text-ink truncate max-w-xs">{row.prompt}</td>
                  <td className="p-4 font-mono text-accent">{row.tier}</td>
                  <td className="p-4 font-mono text-ink">{row.model}</td>
                  <td className="p-4 text-right font-mono text-warn">{row.cost_usd == null ? 'Unavailable' : `$${Number(row.cost_usd).toFixed(5)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// 5. Logs Screen
// ─────────────────────────────────────────────────────────────────────────────
const LogsScreen = () => {
  const [logs, setLogs]         = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [selected, setSelected] = useState(null);
  const [search, setSearch]     = useState('');

  useEffect(() => {
    apiFetch('/api/logs?limit=200')
      .then(setLogs)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const filtered = search
    ? logs.filter(l =>
        l.prompt.toLowerCase().includes(search.toLowerCase()) ||
        l.model.toLowerCase().includes(search.toLowerCase()) ||
        l.tier.toLowerCase().includes(search.toLowerCase())
      )
    : logs;

  return (
    <div className="pb-12">
      <div className="bg-surface-raised border border-line rounded-xl overflow-hidden mb-6">
        <div className="p-4 border-b border-line flex items-center space-x-4">
          <div className="relative flex-1 max-w-xs">
            <div className="absolute left-3 top-2.5 text-ink-faint">{Icons.Search}</div>
            <input
              type="text"
              placeholder="Search logs…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full bg-surface border border-line rounded-lg py-2 pl-10 pr-3 text-sm text-ink outline-none focus:border-accent/50"
            />
          </div>
          {logs.length > 0 && (
            <span className="text-xs text-ink-faint">{filtered.length} records</span>
          )}
        </div>

        <DelayedLoader loading={loading} label="Loading inference records…" className="p-6" />
        {error   && <div className="p-6"><ApiError msg={error} /></div>}

        {!loading && !error && logs.length === 0 && (
          <EmptyState message="No inference records yet. Records appear here when prompts are executed through the Theriad pipeline." />
        )}

        {!loading && !error && filtered.length > 0 && (
          <table className="w-full text-sm">
            <thead className="bg-surface text-ink-faint text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left p-3 font-medium">Timestamp</th>
                <th className="text-left p-3 font-medium">Prompt Preview</th>
                <th className="text-left p-3 font-medium">Tier</th>
                <th className="text-left p-3 font-medium">Model</th>
                <th className="text-right p-3 font-medium">Cost</th>
                <th className="text-right p-3 font-medium">Latency</th>
                <th className="text-center p-3 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {filtered.map(log => (
                <tr key={log.id} onClick={() => setSelected(log)} className="hover:bg-surface-hover cursor-pointer">
                  <td className="p-3 text-ink-muted font-mono text-xs">{formatLogTimestamp(log.timestamp ?? log.time)}</td>
                  <td className="p-3 text-ink truncate max-w-xs">{log.prompt}</td>
                  <td className="p-3 font-mono text-accent">{log.tier}</td>
                  <td className="p-3 font-mono text-ink">{log.model}</td>
                  <td className="p-3 text-right font-mono text-ink">{log.cost}</td>
                  <td className="p-3 text-right text-ink-muted">{log.latency}</td>
                  <td className="p-3 text-center">
                    <Badge color={log.status === 'Routed' ? 'emerald' : 'amber'}>{log.status}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Detail drawer */}
      {selected && (
        <div
          className="fixed inset-0 bg-black/40 z-40 flex justify-end backdrop-blur-sm"
          onClick={() => setSelected(null)}
        >
          <div
            className="w-full max-w-2xl bg-surface-hover h-full overflow-y-auto border-l border-line p-8"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-ink">Routing Record</h2>
              <button onClick={() => setSelected(null)} className="text-ink-muted hover:text-ink text-xl">✕</button>
            </div>

            <div className="mb-6">
              <p className="text-xs text-ink-faint uppercase mb-1">Prompt / Routing Reason</p>
              <p className="text-sm text-ink bg-surface p-3 rounded border border-line">{selected.prompt}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div><p className="text-xs text-ink-faint">Model</p><p className="text-sm font-mono text-accent mt-1">{selected.model}</p></div>
              <div><p className="text-xs text-ink-faint">Cost</p><p className="text-sm font-mono text-ink mt-1">{selected.cost}</p></div>
              <div><p className="text-xs text-ink-faint">Tokens</p><p className="text-sm font-mono text-ink mt-1">{selected.tokens}</p></div>
            </div>

            <div className="flex space-x-2 mb-8">
              <button
                onClick={() => navigator.clipboard?.writeText(JSON.stringify(selected, null, 2))}
                className="bg-surface-hover hover:bg-line-strong text-ink px-3 py-1.5 rounded text-xs font-medium flex items-center space-x-1"
              >
                <span>{Icons.Copy}</span>
                <span>Copy JSON</span>
              </button>
            </div>

            <div className="bg-surface p-4 rounded-lg border border-line">
              <p className="text-xs text-ink-faint uppercase tracking-wider mb-3">Raw JSON Record</p>
              <pre className="text-xs text-ok font-mono overflow-x-auto">
                {JSON.stringify(selected, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Browser Tools / MCP status card (used inside SettingsScreen)
// ─────────────────────────────────────────────────────────────────────────────
function BrowserBackendSetup({ currentBackend, options, onSaved }) {
  const backendOptions = [
    {
      value: 'playwright',
      label: 'Playwright',
      badge: 'Recommended',
      desc: 'Free headless Chromium, runs locally. No API key needed.',
    },
    {
      value: 'playwright_mcp',
      label: 'Playwright MCP',
      badge: 'Node.js',
      desc: 'Playwright via the official @playwright/mcp server. Requires Node.js.',
    },
    {
      value: 'browserbase',
      label: 'BrowserBase',
      badge: 'Cloud',
      desc: 'Managed cloud browser. Get a free API key at browserbase.com.',
    },
    {
      value: 'electron',
      label: 'Electron Desktop',
      badge: 'App',
      desc: 'Connects to the Theriad desktop app (must be running on port 47200).',
    },
    {
      value: 'chrome_mcp',
      label: 'Chrome DevTools',
      badge: 'Extension',
      desc: 'Connects to your Chrome browser via DevTools MCP. Requires server-side configuration.',
    },
  ];

  const initial = (currentBackend && currentBackend !== 'mock') ? currentBackend : 'playwright';
  const [selected, setSelected] = useState(initial);
  const [bbKey, setBbKey]       = useState('');
  const [saving, setSaving]     = useState(false);
  const [result, setResult]     = useState(null);

  const handleSave = async () => {
    setSaving(true);
    setResult(null);
    try {
      const body = { backend: selected };
      if (selected === 'browserbase' && bbKey.trim()) body.browserbase_api_key = bbKey.trim();
      const res = await apiFetch('/api/settings/browser-backend', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      if (res.available) {
        setResult({ ok: true, message: `${selected} is active and ready.` });
        onSaved?.();
      } else {
        setResult({ ok: false, message: res.error || `Saved, but ${selected} is not available yet.` });
        onSaved?.();
      }
    } catch (e) {
      setResult({ ok: false, message: e.message || 'Save failed.' });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <p className="text-sm font-medium text-ink mb-1">Backend override</p>
      <p className="text-xs text-ink-faint mb-3">
        Automatic is the normal path. Choose a specific backend only when diagnosing an installation.
      </p>
      <div className="space-y-2 mb-3">
        {backendOptions.map(opt => {
          const info = options?.[opt.value] ?? {};
          const active = selected === opt.value;
          return (
            <label
              key={opt.value}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                active ? 'border-accent bg-accent/5' : 'border-line hover:border-line-strong'
              }`}
            >
              <input
                type="radio"
                name="browser-backend"
                value={opt.value}
                checked={active}
                onChange={() => { setSelected(opt.value); setResult(null); }}
                className="mt-0.5"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium text-ink">{opt.label}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-surface-hover text-ink-muted">
                    {opt.badge}
                  </span>
                  {opt.value === currentBackend && currentBackend !== 'mock' && (
                    <span className="text-xs text-ok">● active</span>
                  )}
                  {opt.value === 'playwright' && info.chromium_installed && (
                    <span className="text-xs text-ok">installed</span>
                  )}
                  {opt.value === 'playwright' && !info.chromium_installed && (
                    <span className="text-xs text-warn">needs install</span>
                  )}
                  {opt.value === 'playwright_mcp' && info.node_available && (
                    <span className="text-xs text-ok">Node.js found</span>
                  )}
                  {opt.value === 'playwright_mcp' && !info.node_available && (
                    <span className="text-xs text-warn">Node.js not found</span>
                  )}
                  {opt.value === 'browserbase' && info.api_key_configured && (
                    <span className="text-xs text-ok">key saved</span>
                  )}
                </div>
                <p className="text-xs text-ink-faint mt-0.5">{opt.desc}</p>
                {opt.value === 'playwright' && active && !info.chromium_installed && (
                  <p className="text-xs text-warn mt-1 font-mono">
                    {info.package_installed
                      ? 'playwright install chromium'
                      : 'pip install playwright && playwright install chromium'}
                  </p>
                )}
                {opt.value === 'playwright_mcp' && active && !info.node_available && (
                  <p className="text-xs text-warn mt-1">
                    Install Node.js from <span className="font-mono">nodejs.org</span>, then restart the server.
                  </p>
                )}
                {opt.value === 'playwright_mcp' && active && info.node_available && (
                  <p className="text-xs text-ink-faint mt-1 font-mono">npx @playwright/mcp  (downloaded on first use)</p>
                )}
                {opt.value === 'browserbase' && active && (
                  <input
                    type="password"
                    placeholder={info.api_key_configured ? '••••••••  (saved — paste to replace)' : 'bb_live_...  API key from browserbase.com/settings'}
                    value={bbKey}
                    onChange={e => setBbKey(e.target.value)}
                    className="mt-2 w-full text-xs font-mono bg-surface border border-line rounded px-2 py-1.5 text-ink placeholder-ink-faint focus:outline-none focus:border-accent"
                  />
                )}
                {opt.value === 'electron' && active && !info.available && (
                  <p className="text-xs text-ink-faint mt-1">
                    Start the Theriad desktop app, then save to connect.
                  </p>
                )}
                {opt.value === 'chrome_mcp' && active && !info.enabled && (
                  <p className="text-xs text-ink-faint mt-1 font-mono">
                    Set THERIAD_CHROME_MCP_ENABLED=true and restart the server.
                  </p>
                )}
              </div>
            </label>
          );
        })}
      </div>
      <button
        onClick={handleSave}
        disabled={saving}
        className="px-4 py-1.5 text-sm font-medium rounded-lg bg-accent text-white hover:bg-accent-hover disabled:opacity-50 transition-colors"
      >
        {saving ? 'Saving…' : 'Save'}
      </button>
      {result && (
        <p className={`mt-2 text-xs ${result.ok ? 'text-ok' : 'text-warn'}`}>
          {result.message}
        </p>
      )}
    </div>
  );
}


function BrowserToolsSection() {
  const [toolsStatus,   setToolsStatus]   = useState(null);
  const [runtimeStatus, setRuntimeStatus] = useState(null);
  const [capabilityManifest, setCapabilityManifest] = useState(null);
  const [backendOptions, setBackendOptions] = useState(null);
  const [loading, setLoading]             = useState(true);
  const [statusError, setStatusError]     = useState(false);

  const loadAll = () => {
    setLoading(true);
    setStatusError(false);
    Promise.all([
      apiFetch('/api/tools/status').catch(() => null),
      apiFetch('/api/runtime/status').catch(() => null),
      apiFetch('/api/agent/capabilities').catch(() => null),
      apiFetch('/api/settings/browser-backend').catch(() => null),
    ]).then(([ts, rs, manifest, bo]) => {
      setToolsStatus(ts);
      setRuntimeStatus(rs);
      setCapabilityManifest(manifest);
      setBackendOptions(bo);
      setStatusError(!ts && !rs && !manifest && !bo);
    }).finally(() => setLoading(false));
  };

  useEffect(() => { loadAll(); }, []);

  const cdp = toolsStatus?.servers?.find(s => s.server_name === 'chrome_devtools');
  const rs  = runtimeStatus;

  const profileLabel = rs?.runtime_profile === 'personal_trusted'
    ? 'Personal Trusted'
    : 'Consumer Secure (default)';
  const profileColor = rs?.runtime_profile === 'personal_trusted'
    ? 'text-warn'
    : 'text-ok';

  const browserModeLabel = {
    ephemeral:            'Ephemeral (isolated)',
    dedicated_persistent: 'Dedicated persistent',
    real_profile:         'Real Chrome profile',
  }[rs?.browser_profile_mode] ?? rs?.browser_profile_mode ?? '—';

  // Action backend label — derived from runtime config (browser_action_backend
  // in to_safe_dict()). "mock" is the safe default; "playwright" means real
  // low-risk actions are enabled via THERIAD_BROWSER_ACTION_BACKEND=playwright.
  const _ab = rs?.browser_action_backend ?? 'mock';
  const currentBackend = backendOptions?.current_backend ?? _ab;
  const currentBackendOption = backendOptions?.options?.[currentBackend];
  const automaticBrowserReady = currentBackend === 'mock'
    ? Boolean(backendOptions?.options?.playwright?.available)
    : Boolean(currentBackendOption?.available);
  const automaticWebSearchReady = Boolean(
    backendOptions && backendOptions.web_search?.available !== false
  );
  const hostedAgentExecutionAllowed = capabilityManifest?.deployment?.hosted_agent_execution_allowed !== false;
  const statusLabel = loading
    ? 'Checking internet access…'
    : statusError
      ? 'Couldn’t check internet access'
      : !hostedAgentExecutionAllowed
        ? 'Web search ready · interactive agents paused'
      : automaticWebSearchReady && automaticBrowserReady
        ? 'Ready for web tasks'
        : automaticWebSearchReady
          ? 'Web search ready'
          : 'Internet access needs setup';
  const statusMessage = loading
    ? 'Theriad is checking the automatic web access it uses for current questions.'
    : statusError
      ? 'Try again in a moment. Your settings are unchanged.'
      : !hostedAgentExecutionAllowed
        ? 'This hosted deployment can answer current-information questions, but browser and terminal actions stay paused until an isolated executor is configured.'
      : automaticWebSearchReady && automaticBrowserReady
        ? 'Theriad will choose the available isolated browser automatically for tasks that need live navigation or interaction.'
        : automaticWebSearchReady
          ? 'Current-information questions work automatically. Interactive tasks that need clicking may need a browser backend installed by a developer.'
          : 'Web access is not available on this installation yet. A developer can finish setup in diagnostics below.';
  const statusColor = loading || statusError || !automaticWebSearchReady || !hostedAgentExecutionAllowed ? 'text-warn' : 'text-ok';
  const statusDot = loading || statusError || !automaticWebSearchReady || !hostedAgentExecutionAllowed ? 'bg-warn' : 'bg-ok';
  const privacyMessage = rs?.browser_profile_mode === 'real_profile'
    ? 'This installation has access to a personal Chrome profile. Keep that mode for trusted, private work only.'
    : 'Theriad sends only the search terms needed for a web question and uses a temporary, isolated browser profile for interactive tasks. Challenge cookies stay with that task; your personal Chrome profile and open tabs are not used.';
  const chromeDot   = cdp?.available ? 'bg-ok'   : 'bg-warn';
  const chromeLabel = cdp == null ? 'Checking…' : cdp.available ? 'Connected' : 'Not configured';
  const chromeColor = cdp?.available ? 'text-ok' : 'text-warn';
  const extensionState = capabilityManifest?.browser?.extension;
  const terminalState = capabilityManifest?.terminal;
  const desktopState = capabilityManifest?.desktop;

  return (
    <div className="bg-surface-raised border border-line rounded-xl p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-10 h-10 shrink-0 rounded-xl bg-accent/10 text-accent flex items-center justify-center">
            {Icons.Globe}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-lg font-semibold text-ink">Internet access</h2>
              <Badge color="slate">Automatic</Badge>
            </div>
            <p className="text-sm text-ink-muted mt-1 max-w-2xl">
              Theriad browses the web when a task needs current information. You do not need to choose a browser or connect your personal Chrome.
            </p>
          </div>
        </div>
        <div className={`w-2.5 h-2.5 rounded-full mt-2 shrink-0 ${statusDot}`} aria-label={statusLabel} />
      </div>

      <div className="mt-6 flex items-start justify-between gap-4 rounded-xl border border-line bg-surface p-4">
        <div className="flex items-start gap-3 min-w-0">
          <div className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${statusDot}`} />
          <div className="min-w-0">
            <p className={`text-sm font-medium ${statusColor}`}>{statusLabel}</p>
            <p className="text-xs text-ink-faint mt-1 max-w-2xl">{statusMessage}</p>
          </div>
        </div>
        {!loading && (
          <button
            type="button"
            onClick={loadAll}
            className="shrink-0 text-xs font-medium text-accent hover:text-accent-hover transition-colors"
          >
            Check again
          </button>
        )}
      </div>

      <p className="mt-4 text-xs text-ink-faint">{privacyMessage}</p>

      <details className="mt-5 border-t border-line pt-4">
        <summary className="cursor-pointer text-sm font-medium text-ink hover:text-accent transition-colors">
          Advanced diagnostics
          <span className="block mt-1 text-xs font-normal text-ink-faint">
            Technical browser status and backend overrides. Most users never need to open this.
          </span>
        </summary>

        {loading ? <DelayedLoader loading={loading} label="Checking technical status…" /> : (
          <div className="space-y-4 pt-4">

          {/* Runtime Profile row */}
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-ink">Runtime Profile</p>
              <p className="text-xs text-ink-faint mt-1">
                <span className={profileColor}>{profileLabel}</span>
              </p>
              {rs?.profile_badge && (
                <p className="text-xs text-ink-faint mt-1 max-w-sm italic">{rs.profile_badge}</p>
              )}
            </div>
            <div className={`w-2.5 h-2.5 rounded-full mt-1 flex-shrink-0 ${
              rs?.runtime_profile === 'personal_trusted' ? 'bg-warn' : 'bg-ok'
            }`}></div>
          </div>

          {/* Browser Profile Mode row */}
          <div className="flex items-start justify-between border-t border-line pt-4">
            <div>
              <p className="text-sm font-medium text-ink">Browser Profile Mode</p>
              <p className="text-xs text-ink-faint mt-1">
                {browserModeLabel}
                {rs?.allow_real_chrome_profile && (
                  <span className="ml-2 text-warn">(real Chrome profile allowed)</span>
                )}
              </p>
            </div>
            <div className={`w-2.5 h-2.5 rounded-full mt-1 flex-shrink-0 ${
              rs?.browser_profile_mode === 'ephemeral' ? 'bg-ok' : 'bg-warn'
            }`}></div>
          </div>

          {/* Chrome MCP row */}
          <div className="flex items-start justify-between border-t border-line pt-4">
            <div>
              <p className="text-sm font-medium text-ink">Chrome DevTools MCP</p>
              <p className="text-xs text-ink-faint mt-1">
                Status:{' '}
                <span className={chromeColor}>{chromeLabel}</span>
              </p>
              {cdp?.error && (
                <p className="text-xs text-ink-faint mt-1 max-w-sm">{cdp.error}</p>
              )}
              {!cdp?.available && (
                <p className="text-xs text-ink-faint mt-1">
                  Set <code className="font-mono">THERIAD_CHROME_MCP_ENABLED=true</code> to enable.
                </p>
              )}
            </div>
            <div className={`w-2.5 h-2.5 rounded-full mt-1 flex-shrink-0 ${chromeDot}`}></div>
          </div>

          {/* Browser Agent Backend — interactive setup. */}
          <div className="border-t border-line pt-4">
            <BrowserBackendSetup
              currentBackend={currentBackend}
              options={backendOptions?.options}
              onSaved={loadAll}
            />
          </div>

          <div className="flex items-start justify-between border-t border-line pt-4">
            <div>
              <p className="text-sm font-medium text-ink">Authenticated Chrome Extension</p>
              <p className="text-xs text-ink-faint mt-1">
                {extensionState?.paired
                  ? `${extensionState.connections?.length || 1} task-bound connection paired`
                  : 'Not paired — create a task pairing code and assign the current tab in the extension.'}
              </p>
            </div>
            <div className={`w-2.5 h-2.5 rounded-full mt-1 ${extensionState?.paired ? 'bg-ok' : 'bg-warn'}`}></div>
          </div>

          <div className="flex items-start justify-between border-t border-line pt-4">
            <div>
              <p className="text-sm font-medium text-ink">Terminal</p>
              <p className="text-xs text-ink-faint mt-1">
                {terminalState?.available ? 'Guarded host execution available — PTY, streaming, stdin, background jobs, and cancellation.' : 'Unavailable'}
              </p>
              {terminalState?.warning && <p className="text-xs text-warn mt-1">{terminalState.warning}</p>}
            </div>
            <div className={`w-2.5 h-2.5 rounded-full mt-1 ${terminalState?.available ? 'bg-ok' : 'bg-warn'}`}></div>
          </div>

          {!hostedAgentExecutionAllowed && (
            <div className="flex items-start justify-between border-t border-line pt-4">
              <div>
                <p className="text-sm font-medium text-ink">Hosted agent execution</p>
                <p className="text-xs text-warn mt-1">
                  Paused until this deployment has an OS-isolated executor. Capability inspection remains available.
                </p>
              </div>
              <div className="w-2.5 h-2.5 rounded-full mt-1 bg-warn"></div>
            </div>
          )}

          <div className="flex items-start justify-between border-t border-line pt-4">
            <div>
              <p className="text-sm font-medium text-ink">macOS Computer Access</p>
              <p className="text-xs text-ink-faint mt-1">
                {desktopState?.available ? 'Enabled with app/window assignment.' : desktopState?.error || desktopState?.setup || 'Disabled.'}
              </p>
            </div>
            <div className={`w-2.5 h-2.5 rounded-full mt-1 ${desktopState?.available ? 'bg-ok' : 'bg-warn'}`}></div>
          </div>

          {/* Subtask allowlists summary */}
          {toolsStatus?.subtask_allowlists?.length > 0 && (
            <div className="border-t border-line pt-4">
              <p className="text-xs text-ink-faint mb-2 uppercase tracking-wider">Subtask allowlists</p>
              <div className="flex flex-wrap gap-1.5">
                {toolsStatus.subtask_allowlists.map(name => (
                  <span key={name} className="text-xs font-mono bg-surface-hover text-ink-muted px-2 py-0.5 rounded">
                    {name}
                  </span>
                ))}
              </div>
            </div>
          )}
          </div>
        )}
      </details>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 6. Settings Screen
// ─────────────────────────────────────────────────────────────────────────────
const DEFAULT_RUNTIME_SETTINGS = {
  regular_chat_returns_answer: true,
  allow_follow_up_questions: true,
  route_only_available: false,
  disable_premium_models: false,
  model_origin_policy: 'any',
  max_cost_per_request_usd: null,
  max_daily_spend_usd: 5.00,
};

function RuntimeToggle({ checked, onChange, disabled, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onChange}
      className={'w-10 h-6 rounded-full p-1 flex items-center transition-colors ' +
        (checked ? 'bg-accent justify-end' : 'bg-surface-hover justify-start') +
        (disabled ? ' opacity-50 cursor-not-allowed' : ' cursor-pointer')}
    >
      <span className={'w-4 h-4 rounded-full ' + (checked ? 'bg-white' : 'bg-ink-faint')} />
    </button>
  );
}

const SettingsScreen = ({ user }) => {
  const [status, setStatus]       = useState(null);
  const [loading, setLoading]     = useState(true);
  const [runtimeSettings, setRuntimeSettings] = useState(DEFAULT_RUNTIME_SETTINGS);
  const [runtimeSettingsLoading, setRuntimeSettingsLoading] = useState(true);
  const [runtimeSaving, setRuntimeSaving] = useState(false);
  const [runtimeError, setRuntimeError] = useState(null);

  useEffect(() => {
    apiFetch('/api/status')
      .then(setStatus)
      .catch(() => setStatus(null))
      .finally(() => setLoading(false));

    apiFetch('/api/settings/runtime')
      .then(setRuntimeSettings)
      .catch(error => setRuntimeError(error.message))
      .finally(() => setRuntimeSettingsLoading(false));
  }, []);

  const ollConnected = status?.ollama?.configured ?? false;

  async function saveRuntimeSettings(patch) {
    const previous = runtimeSettings;
    setRuntimeSettings(current => ({ ...current, ...patch }));
    setRuntimeSaving(true);
    setRuntimeError(null);
    try {
      const saved = await apiFetch('/api/settings/runtime', {
        method: 'PUT',
        body: JSON.stringify(patch),
      });
      setRuntimeSettings(saved);
    } catch (error) {
      setRuntimeSettings(previous);
      setRuntimeError(error.message);
    } finally {
      setRuntimeSaving(false);
    }
  }

  function commitRuntimeLimit(key) {
    const raw = runtimeSettings[key];
    const value = raw === '' || raw == null ? null : Number(raw);
    if (value !== null && (!Number.isFinite(value) || value < 0)) {
      setRuntimeError('Limits must be non-negative numbers.');
      return;
    }
    saveRuntimeSettings({ [key]: value });
  }

  return (
    <div className="theriad-settings-page pb-12">
      <div className="theriad-settings-sections space-y-8">
        <ProviderSettings />

        {/* Local runtime */}
        <div className="theriad-settings-card bg-surface-raised border border-line rounded-xl p-6">
          <h2 className="text-lg font-semibold text-ink mb-4">Local runtime</h2>
          {loading ? <DelayedLoader loading={loading} className="py-4" /> : (
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm font-medium text-ink">Ollama Local</p>
                <p className="text-xs text-ink-faint mt-1">
                  Status:{' '}
                  <span className={ollConnected ? 'text-ok' : 'text-ink-faint'}>
                    {ollConnected ? 'Configured' : 'Not configured'}
                  </span>
                  {' '}• URL:{' '}
                  <span className="font-mono">{status?.ollama?.url ?? 'localhost:11434'}</span>
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Safety — persisted account controls */}
        <div className="theriad-settings-card bg-surface-raised border border-line rounded-xl p-6">
          <div className="flex items-center justify-between gap-4 mb-4">
            <h2 className="text-lg font-semibold text-ink">Safety &amp; Limits</h2>
            <Badge color={runtimeSaving ? 'slate' : 'emerald'}>
              {runtimeSaving ? 'Saving…' : 'Live controls'}
            </Badge>
          </div>
          {runtimeError && (
            <p className="mb-4 text-xs text-warn bg-warn/10 border border-warn/20 rounded-lg px-3 py-2">
              {runtimeError}
            </p>
          )}
          <div className="space-y-5">
            <div className="theriad-setting-row flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Regular chat returns an answer</p>
                <p className="text-xs text-ink-faint mt-1">Messages run immediately through the active assistant connection.</p>
              </div>
              <RuntimeToggle
                checked={runtimeSettings.regular_chat_returns_answer}
                onChange={() => saveRuntimeSettings({
                  regular_chat_returns_answer: !runtimeSettings.regular_chat_returns_answer,
                })}
                disabled={runtimeSettingsLoading || runtimeSaving}
                label="Regular chat returns an answer"
              />
            </div>
            <div className="theriad-setting-row flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Allow follow-up questions</p>
                <p className="text-xs text-ink-faint mt-1">When a request is materially ambiguous, let the assistant ask one concise question instead of guessing.</p>
              </div>
              <RuntimeToggle
                checked={runtimeSettings.allow_follow_up_questions}
                onChange={() => saveRuntimeSettings({
                  allow_follow_up_questions: !runtimeSettings.allow_follow_up_questions,
                })}
                disabled={runtimeSettingsLoading || runtimeSaving}
                label="Allow follow-up questions"
              />
            </div>
            <details className="rounded-lg border border-line bg-surface-hover/40 px-4">
              <summary className="cursor-pointer list-none py-3 text-sm font-medium text-ink hover:text-accent transition-colors">
                Advanced
                <span className="block mt-1 text-xs font-normal text-ink-faint">
                  Routing, model selection, and per-request controls.
                </span>
              </summary>
              <div className="pb-4 space-y-5">
            <div className="theriad-setting-row flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Route-only mode remains available</p>
                <p className="text-xs text-ink-faint mt-1">Use it when you want to inspect model selection without generating an answer.</p>
              </div>
              <RuntimeToggle
                checked={runtimeSettings.route_only_available}
                onChange={() => saveRuntimeSettings({
                  route_only_available: !runtimeSettings.route_only_available,
                })}
                disabled={runtimeSettingsLoading || runtimeSaving}
                label="Route-only mode remains available"
              />
            </div>
            <div className="theriad-setting-row flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Disable premium models by default</p>
                <p className="text-xs text-ink-faint mt-1">Exclude <span className="font-mono">premium</span> tier from routing unless manually specified.</p>
              </div>
              <RuntimeToggle
                checked={runtimeSettings.disable_premium_models}
                onChange={() => saveRuntimeSettings({
                  disable_premium_models: !runtimeSettings.disable_premium_models,
                })}
                disabled={runtimeSettingsLoading || runtimeSaving}
                label="Disable premium models by default"
              />
            </div>
            <div className="theriad-setting-row flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Model origin policy</p>
                <p className="text-xs text-ink-faint mt-1">Limit Theriad routing to models from cataloged US vendors.</p>
              </div>
              <select
                value={runtimeSettings.model_origin_policy}
                onChange={event => saveRuntimeSettings({ model_origin_policy: event.target.value })}
                disabled={runtimeSettingsLoading || runtimeSaving}
                aria-label="Model origin policy"
                className="shrink-0 rounded-lg border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <option value="any">Any origin</option>
                <option value="us_only">US-only models</option>
              </select>
            </div>
            <div className="theriad-setting-limits pt-4 border-t border-line">
              <div>
                <label className="block text-sm font-medium text-ink mb-1">Max Cost per Request ($)</label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={runtimeSettings.max_cost_per_request_usd ?? ''}
                  onChange={event => setRuntimeSettings(current => ({
                    ...current,
                    max_cost_per_request_usd: event.target.value,
                  }))}
                  onBlur={() => commitRuntimeLimit('max_cost_per_request_usd')}
                  disabled={runtimeSettingsLoading || runtimeSaving}
                  className="w-full bg-surface border border-line rounded-lg py-2 px-3 text-sm text-ink outline-none focus:border-accent/50 font-mono"
                />
              </div>
            </div>
              </div>
            </details>
            <div className="theriad-setting-limits pt-4 border-t border-line">
              <div>
                <label className="block text-sm font-medium text-ink mb-1">Max Daily Spend ($)</label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={runtimeSettings.max_daily_spend_usd ?? ''}
                  onChange={event => setRuntimeSettings(current => ({
                    ...current,
                    max_daily_spend_usd: event.target.value,
                  }))}
                  onBlur={() => commitRuntimeLimit('max_daily_spend_usd')}
                  disabled={runtimeSettingsLoading || runtimeSaving}
                  className="w-full bg-surface border border-line rounded-lg py-2 px-3 text-sm text-ink outline-none focus:border-accent/50 font-mono"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Internet access is automatic for normal users; technical controls
            remain available under the component's advanced disclosure. */}
        <BrowserToolsSection />

      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Agent Screen — Browser-agent human-in-the-loop panel (MOCK mode)
// ─────────────────────────────────────────────────────────────────────────────

const AGENT_RISK_COLOR = {
  low:      'text-ok',
  medium:   'text-warn',
  high:     'text-danger',
  critical: 'text-danger',
};

function LocalBrowserPanel({ agentSessionId = null }) {
  const [ownedSessionId, setOwnedSessionId] = useState(null);
  const [visible, setVisible] = useState(false);
  const [snapshot, setSnapshot] = useState(null);
  const [browserError, setBrowserError] = useState(null);
  const [busy, setBusy] = useState(false);
  const sessionId = agentSessionId || ownedSessionId;

  async function refresh(targetId = sessionId) {
    if (!targetId) return;
    try {
      const data = await apiFetch(
        `/api/browser/observe?session_id=${encodeURIComponent(targetId)}&screenshot=true`
      );
      setSnapshot(data);
      setBrowserError(null);
    } catch (e) {
      setBrowserError(friendlyError(e.message));
    }
  }

  useEffect(() => {
    if (agentSessionId) refresh(agentSessionId);
  }, [agentSessionId]);

  async function startLocalBrowser() {
    setBusy(true);
    setBrowserError(null);
    try {
      const data = await apiFetch('/api/browser/session/start', {
        method: 'POST',
        body: JSON.stringify({ visible }),
      });
      setOwnedSessionId(data.session_id);
      await refresh(data.session_id);
    } catch (e) {
      setBrowserError(friendlyError(e.message));
    } finally {
      setBusy(false);
    }
  }

  async function stopLocalBrowser() {
    if (!sessionId) return;
    setBusy(true);
    try {
      await apiFetch('/api/browser/session/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      });
      setOwnedSessionId(null);
      setSnapshot(null);
    } catch (e) {
      setBrowserError(friendlyError(e.message));
    } finally {
      setBusy(false);
    }
  }

  async function resolvePending(approved) {
    if (!sessionId) return;
    setBusy(true);
    try {
      await apiFetch('/api/browser/action', {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          approved,
          deny: !approved,
        }),
      });
      await refresh(sessionId);
    } catch (e) {
      setBrowserError(friendlyError(e.message));
    } finally {
      setBusy(false);
    }
  }

  const pending = snapshot?.pending_action;
  const log = snapshot?.action_log ?? [];

  return (
    <section className="bg-surface-sunken border border-line rounded-xl overflow-hidden">
      <header className="px-5 py-3 border-b border-line flex items-center gap-3">
        <div>
          <h3 className="text-sm font-semibold text-ink">Local Chromium</h3>
          <p className="text-xs text-ink-faint">
            Isolated Playwright session · bundled Chromium
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {!agentSessionId && !ownedSessionId && (
            <label className="flex items-center gap-1.5 text-xs text-ink-muted">
              <input
                type="checkbox"
                checked={visible}
                onChange={e => setVisible(e.target.checked)}
                className="accent-accent"
              />
              Visible browser
            </label>
          )}
          {sessionId ? (
            <>
              <button
                onClick={() => refresh()}
                disabled={busy}
                className="px-3 py-1.5 text-xs rounded-lg border border-line text-ink-muted hover:text-ink"
              >
                Refresh
              </button>
              {!agentSessionId && (
                <button
                  onClick={stopLocalBrowser}
                  disabled={busy}
                  className="px-3 py-1.5 text-xs rounded-lg border border-danger/30 text-danger bg-danger/10"
                >
                  Stop
                </button>
              )}
            </>
          ) : (
            <button
              onClick={startLocalBrowser}
              disabled={busy}
              className="px-3 py-1.5 text-xs rounded-lg border border-accent/30 text-accent bg-accent/10"
            >
              {busy ? 'Starting…' : 'Start browser'}
            </button>
          )}
        </div>
      </header>

      {browserError && <div className="m-4"><ApiError msg={browserError} /></div>}

      <div className="grid lg:grid-cols-[minmax(0,1fr)_280px] min-h-56">
        <div className="p-4 border-b lg:border-b-0 lg:border-r border-line min-w-0">
          <p className="text-[10px] uppercase tracking-wider text-ink-faint mb-1">Current URL</p>
          <p className="text-xs font-mono text-ink-muted truncate mb-3">
            {snapshot?.current_url || (sessionId ? 'Loading…' : 'No active browser session')}
          </p>
          {snapshot?.original_user_prompt && (
            <div className="mb-3 text-xs">
              <p className="text-ink-faint">Original prompt</p>
              <p className="text-ink">{snapshot.original_user_prompt}</p>
              <p className="text-ink-faint mt-1">Resolved initial URL</p>
              <p className="font-mono text-ink-muted break-all">
                {snapshot.resolved_initial_url || snapshot.current_url}
              </p>
              {snapshot.navigation_reason && (
                <p className="mt-1 text-warn">{snapshot.navigation_reason}</p>
              )}
            </div>
          )}
          {snapshot?.screenshot_base64 ? (
            <img
              src={`data:image/png;base64,${snapshot.screenshot_base64}`}
              alt="Current Chromium page"
              className="w-full max-h-72 object-contain object-top rounded-lg border border-line bg-white"
            />
          ) : (
            <div className="h-40 rounded-lg border border-dashed border-line flex items-center justify-center text-xs text-ink-faint">
              Screenshot preview appears here.
            </div>
          )}
        </div>
        <div className="p-4 space-y-4 overflow-y-auto max-h-96">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">Proposed next action</p>
            {pending ? (
              <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 space-y-2">
                <pre className="text-xs text-ink whitespace-pre-wrap">{JSON.stringify(pending, null, 2)}</pre>
                <div className="flex gap-2">
                  <button onClick={() => resolvePending(true)}
                    className="px-2.5 py-1 text-xs rounded border border-ok/30 text-ok bg-ok/10">
                    Approve
                  </button>
                  <button onClick={() => resolvePending(false)}
                    className="px-2.5 py-1 text-xs rounded border border-danger/30 text-danger bg-danger/10">
                    Deny
                  </button>
                </div>
              </div>
            ) : (
              <p className="text-xs text-ink-faint">No sensitive action is awaiting approval.</p>
            )}
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-ink-faint mb-2">Action log</p>
            <div className="space-y-1.5">
              {log.length === 0 && <p className="text-xs text-ink-faint">No browser actions yet.</p>}
              {log.slice().reverse().slice(0, 12).map((entry, index) => (
                <div key={index} className="text-[11px] border-b border-line pb-1.5">
                  <span className="font-mono text-ink">{entry.action}</span>
                  <span className="ml-2 text-ink-faint">{entry.status}</span>
                  {entry.url && <p className="font-mono text-ink-faint truncate">{entry.url}</p>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// Human-readable phrasing for the pending-approval card, e.g.
// "Theriad wants to log in on example.com." (Part 6 example copy).
const AGENT_ACTION_PHRASES = {
  login:                 'log in',
  submit_form:           'submit a form',
  send_message:          'send a message',
  purchase:              'make a purchase',
  checkout:              'check out',
  delete:                'delete data',
  delete_account_data:   'delete account data',
  edit_account_data:     'change account settings',
  upload_file:           'upload a file',
  upload_sensitive_file: 'upload a sensitive file',
  download_file:         'download a file',
  enter_sensitive_info:  'enter sensitive information',
  grant_permission:      'grant a new permission',
  access_financial_portal: 'access a financial portal',
  access_medical_portal:   'access a medical portal',
  access_legal_portal:     'access a legal portal',
  install_software:     'install software',
  expose_secrets:        'expose secrets or API keys',
  trade:                 'place a trade',
  invest:                'make an investment',
};

function describeAgentAction(actionType) {
  if (!actionType) return 'take an action';
  return AGENT_ACTION_PHRASES[actionType] ?? actionType.replace(/_/g, ' ');
}

const AGENT_STATUS_COLOR = {
  idle:                 'text-ink-muted',
  running:              'text-accent',
  waiting_for_approval: 'text-warn',
  needs_approval:       'text-warn',
  // D-045: distinct pause state — the action cannot proceed without input
  // (e.g. login credentials) that this release has no capture flow for yet.
  waiting_for_input:    'text-warn',
  completed:            'text-ok',
  failed:               'text-danger',
  stopped:              'text-ink-muted',
  stopped_budget_exceeded: 'text-warn',
};

const AGENT_RESULT_COLOR = {
  success:              'text-ok',
  blocked:              'text-danger',
  approval_pending:     'text-warn',
  error:                'text-danger',
  // D-045 result values
  executed:             'text-ok',
  blocked_missing_input: 'text-warn',
  failed:               'text-danger',
};

// Fake action log entries cycled through during mock steps
const MOCK_STEP_TEMPLATES = [
  { action_type: 'navigate',    result: 'success', summary: 'Navigated to start page' },
  { action_type: 'observe_dom', result: 'success', summary: 'Scanned page structure (mock)' },
  { action_type: 'extract',     result: 'success', summary: 'Extracted mock result set' },
  { action_type: 'scroll',      result: 'success', summary: 'Scrolled to next section' },
  { action_type: 'click',       result: 'success', summary: 'Clicked mock target element' },
];

function makeMockSessionId() {
  return 'mock-' + Math.random().toString(36).slice(2, 10);
}

function makeMockSession(goal, startUrl, policy) {
  return {
    session_id:          makeMockSessionId(),
    goal:                goal,
    status:              'idle',
    current_url:         startUrl || null,
    steps_taken:         0,
    model_calls:         0,
    estimated_cost_usd:  0,
    elapsed_seconds:     0,
    action_log:          [],
    pending_approval:    null,
    final_result:        null,
    error:               null,
    step_diagnostics:    [],
    _mock:               true,       // sentinel: never sent to backend
    _start_ts:           Date.now(),
    _policy:             policy,
  };
}

// Legacy browser runtime panel retained for internal debugging.
function AgentScreen({ incomingSession }) {
  const [goal, setGoal]                       = useState('');
  const [startUrl, setStartUrl]               = useState('');
  const [policy, setPolicy]                   = useState('balanced');
  // Keep the wire fields at their existing defaults while routing stays automatic.
  const speedPreference = 50;
  const qualityPreference = 50;
  const [maxSteps, setMaxSteps]               = useState(20);
  const [maxCost, setMaxCost]                 = useState('');
  const [realObsEnabled, setRealObsEnabled]     = useState(false);
  const [realActionEnabled, setRealActionEnabled] = useState(false);
  // The normal path is automatic.  A concrete backend is only a developer
  // override inside Advanced options.
  const [browserBackend, setBrowserBackend]         = useState('automatic');
  // Permission policy (D-044). "Only Ask if Risk is High" is the recommended
  // default for both consumer_secure and personal_trusted (Part 6) — the
  // user can change it per session.
  const [permissionMode, setPermissionMode]   = useState('ask_high_risk');

  const [sessionId, setSessionId]   = useState(null);
  const [status, setStatus]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  // Start response may include an observation_warning (e.g. Playwright missing)
  const [obsWarning, setObsWarning] = useState(null);

  // Capabilities from backend (Playwright installed, etc.)
  const [caps, setCaps] = useState(null);

  // Runtime profile from backend
  const [runtimeStatus, setRuntimeStatus] = useState(null);

  const TERMINAL = new Set(['completed', 'failed', 'stopped', 'stopped_budget_exceeded']);

  // ── Load capabilities and runtime profile on mount ───────────────────────
  useEffect(() => {
    apiFetch('/api/agent/browser/capabilities')
      .then(data => {
        setCaps(data);
        // The server owns backend selection. A concrete backend is only
        // surfaced as an optional developer override.
        const preferred = data.available_backends?.find(name => name !== 'extension') || 'automatic';
        setBrowserBackend(preferred);
        setRealObsEnabled(preferred !== 'automatic' && preferred !== 'mock');
        setRealActionEnabled(preferred !== 'automatic' && preferred !== 'mock' && (data.real_actions_supported ?? data.real_actions_enabled));
      })
      .catch(() => setCaps({ playwright_available: false, real_observation_supported: false }));
    apiFetch('/api/runtime/status')
      .then(setRuntimeStatus)
      .catch(() => setRuntimeStatus(null));
  }, []);

  // ── Receive incoming session from Chat recommendation card ────────────────
  useEffect(() => {
    if (!incomingSession) return;
    setGoal(incomingSession.goal ?? '');
    setStartUrl(incomingSession.start_url ?? '');
    if (incomingSession.policy) setPolicy(incomingSession.policy);
    setSessionId(null);
    setStatus(null);
    setError(null);
    setObsWarning(null);
  }, [incomingSession]);

  // ── Start session — runs autonomously server-side until pause/terminal
  //    (D-044, Part 2/5). The /start response already carries full status,
  //    so no separate follow-up /status fetch is needed. ───────────────────
  const handleStart = async () => {
    if (!goal.trim()) return;
    setLoading(true);
    setError(null);
    setObsWarning(null);
    setStatus(null);
    setSessionId(null);
    try {
      const body = {
        goal: goal.trim(),
        policy,
        speed_preference: speedPreference,
        quality_preference: qualityPreference,
        max_steps: maxSteps,
        allow_paid_calls: false,
        permission_mode: permissionMode,
        real_observation_enabled: browserBackend !== 'automatic' && realObsEnabled && !!(caps?.playwright_available),
        // Keyed off *supported* (Playwright + Chromium present), not the
        // server-side default backend env setting — the toggle is an explicit
        // per-session opt-in the backend honors when Playwright is installed.
        real_action_enabled: browserBackend !== 'automatic' && realActionEnabled && !!(caps?.real_actions_supported ?? caps?.real_actions_enabled),
      };
      if (browserBackend !== 'automatic') body.browser_backend = browserBackend;
      if (startUrl.trim()) body.start_url = startUrl.trim();
      if (maxCost.trim()) body.max_cost_usd = parseFloat(maxCost);

      const data = await apiFetch('/api/agent/browser/start', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      setSessionId(data.session_id);
      if (data.observation_warning) setObsWarning(data.observation_warning);
      setStatus(data);
    } catch (e) {
      setError(friendlyError(e.message));
    } finally {
      setLoading(false);
    }
  };

  // ── Developer/debug only: manually advance one internal step ─────────────
  // Not exposed in normal UI (Part 2) — kept for a future debug-mode toggle.
  const handleDebugStep = async (steps = 1) => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch('/api/agent/browser/step', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, steps }),
      });
      setStatus(data);
    } catch (e) {
      setError(friendlyError(e.message));
    } finally {
      setLoading(false);
    }
  };

  // ── Stop ──────────────────────────────────────────────────────────────────
  const handleStop = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const data = await apiFetch('/api/agent/browser/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      });
      setStatus(data);
    } catch (e) {
      setError(friendlyError(e.message));
    } finally {
      setLoading(false);
    }
  };

  const handleCaptchaHandoff = async (endpoint) => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch(endpoint, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      });
      setStatus(data);
    } catch (e) {
      setError(friendlyError(e.message));
    } finally {
      setLoading(false);
    }
  };

  // ── Approve / deny pending action — each resumes the autonomous loop
  //    server-side (D-044, Part 1 items 5-6) and returns the next natural
  //    stopping point in the same response. ────────────────────────────────
  const handleApprove = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const data = await apiFetch('/api/agent/browser/approve', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, approved: true }),
      });
      setStatus(data);
    } catch (e) {
      setError(friendlyError(e.message));
    } finally {
      setLoading(false);
    }
  };

  const handleDeny = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const data = await apiFetch('/api/agent/browser/deny', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      });
      setStatus(data);
    } catch (e) {
      setError(friendlyError(e.message));
    } finally {
      setLoading(false);
    }
  };

  const isRunning  = status?.status === 'running';
  const isWaiting  = status?.status === 'waiting_for_approval';
  // D-045: distinct from isWaiting — no pending yes/no decision here, the
  // action cannot proceed without input (credentials) this release cannot
  // capture yet. See STATUS_WAITING_FOR_INPUT in browser_schemas.py.
  const isWaitingForInput = status?.status === 'waiting_for_input';
  const isCaptchaPause = status?.status === 'needs_approval' && status?.captcha_detected;
  const isTerminal = status ? TERMINAL.has(status.status) : false;
  const hasSession = !!sessionId;
  const statusColor = AGENT_STATUS_COLOR[status?.status] ?? 'text-ink-muted';
  // playwright_available = Python package importable
  // playwright_chromium_installed = binaries present (required to actually run)
  const playwrightAvailable   = caps?.playwright_chromium_installed ?? caps?.playwright_available ?? false;
  // "supported" = Playwright + Chromium installed (per-session opt-in works);
  // caps.real_actions_enabled (server default backend) kept as fallback for
  // older backends that don't send real_actions_supported.
  const realActionsSupported  = caps?.real_actions_supported ?? caps?.real_actions_enabled ?? false;
  // Prefer backend truth from session status; fall back to pre-session local state.
  const selectedLiveBackend = (
    browserBackend !== 'mock' && browserBackend !== 'automatic'
  ) ? browserBackend : null;
  const obsBackend      = status?.observation_backend ?? selectedLiveBackend ?? (realObsEnabled && playwrightAvailable ? 'playwright' : 'mock');
  const actionBackend   = status?.action_backend      ?? selectedLiveBackend ?? (realActionEnabled && realActionsSupported ? 'playwright' : 'mock');

  // Publish the effective backends so the Sidebar "Browser:" status reads
  // backend-reported state instead of a hardcoded label.
  useEffect(() => {
    browserBackendStore.set({ observation_backend: obsBackend, action_backend: actionBackend });
  }, [obsBackend, actionBackend]);
  const obsActive       = obsBackend !== 'mock';
  const realActionsActive = actionBackend !== 'mock';
  // Keep backward-compat alias for references that still use the old name.
  const realObsActive   = obsActive;

  return (
    <div className="max-w-4xl mx-auto space-y-6">

      <details className="rounded-xl border border-line bg-surface-raised overflow-hidden">
        <summary className="cursor-pointer px-5 py-3 text-xs font-medium text-ink-muted hover:bg-surface-hover transition-colors">
          Advanced runtime status
        </summary>
        <div className="p-4 space-y-3">
      {/* ── Mode banner ───────────────────────────────────────────────────── */}
      {obsActive && realActionsActive ? (
        <div className="flex items-center gap-3 bg-ok/10 border border-ok/30 rounded-xl px-5 py-3">
          <span className="w-2 h-2 rounded-full bg-ok flex-shrink-0"></span>
          <div className="text-xs text-ok">
            <span className="font-semibold">Live observation and low-risk browser actions are enabled through {actionBackend}.</span>
            <span className="text-ok/70"> Navigation, scrolling, and ordinary link clicks run for real. </span>
            <span className="font-semibold">Sensitive actions still require approval.</span>
          </div>
          <span className="ml-auto flex-shrink-0 text-xs font-mono text-ok whitespace-nowrap">{actionBackend}: live</span>
        </div>
      ) : obsActive ? (
        <div className="flex items-center gap-3 bg-accent/10 border border-accent/30 rounded-xl px-5 py-3">
          <span className="w-2 h-2 rounded-full bg-accent flex-shrink-0"></span>
          <div className="text-xs text-accent">
            <span className="font-semibold">Live observation is enabled through {obsBackend}.</span>
            <span className="text-accent/70"> The selected backend reads the live page. </span>
            <span className="font-semibold text-warn">Browser actions are mocked — no clicks, typing, or form submissions will occur.</span>
          </div>
          <span className="ml-auto flex-shrink-0 text-xs font-mono text-accent whitespace-nowrap">{obsBackend}: live</span>
        </div>
      ) : (
        <div className="flex items-start gap-3 bg-surface-hover border border-line rounded-xl px-5 py-3">
          <span className="w-2 h-2 rounded-full bg-ink-faint flex-shrink-0 mt-1"></span>
          <div className="text-xs text-ink-muted space-y-0.5">
            <p className="font-semibold text-ink">Mock mode — no real browser launched. Observation and actions are both mocked.</p>
            <p>
              Synthetic page data is used. Enable <span className="text-ink">Live Observation</span> below
              {playwrightAvailable
                ? ' to observe live webpages via Playwright.'
                : ' — requires Playwright: '}
              {!playwrightAvailable && <code className="font-mono text-warn">pip install playwright && playwright install chromium</code>}
            </p>
          </div>
          <span className="ml-auto flex-shrink-0 text-xs font-mono text-ink-faint whitespace-nowrap">
            Playwright: {playwrightAvailable ? 'available' : 'not installed'}
          </span>
        </div>
      )}

      {/* ── Runtime profile + Chrome MCP status badges ───────────────────────
           Five distinct categories, kept deliberately separate so "observation"
           (what the agent can see) is never conflated with "actions" (what the
           agent can do). See DECISIONS.md D-039/D-040/D-041/D-042. ────────── */}
      {runtimeStatus && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Runtime */}
          <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium ${
            runtimeStatus.runtime_profile === 'personal_trusted'
              ? 'bg-warn/10 border-warn/30 text-warn'
              : 'bg-ok/10 border-ok/30 text-ok'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              runtimeStatus.runtime_profile === 'personal_trusted' ? 'bg-warn' : 'bg-ok'
            }`}></span>
            Runtime: {runtimeStatus.runtime_profile === 'personal_trusted'
              ? 'Personal Trusted Mode'
              : 'Secure Mode'}
          </span>

          {/* Browser Profile */}
          <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-line bg-surface-hover text-ink-muted font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-ink-faint flex-shrink-0"></span>
            Browser Profile: {runtimeStatus.browser_profile_mode === 'ephemeral'
              ? 'Ephemeral'
              : runtimeStatus.browser_profile_mode === 'dedicated_persistent'
              ? 'Dedicated'
              : 'Real Profile'}
          </span>

          {/* Observation — what the agent can currently see. */}
          <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium ${
            obsActive
              ? 'bg-ok/10 border-ok/30 text-ok'
              : 'bg-surface-hover border-line text-ink-muted'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              obsActive ? 'bg-ok' : 'bg-ink-faint'
            }`}></span>
            Observation: {obsBackend === 'playwright' ? 'Playwright Live'
              : obsBackend === 'chrome_mcp' ? 'Chrome MCP'
              : 'Mock'}
          </span>

          {/* Actions — reflects actual backend; real low-risk actions enabled
              when THERIAD_BROWSER_ACTION_BACKEND=playwright. */}
          <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium ${
            actionBackend === 'playwright' ? 'bg-ok/10 border-ok/30 text-ok'
            : actionBackend === 'chrome_mcp' ? 'bg-info/10 border-info/30 text-info'
            : 'bg-warn/10 border-warn/30 text-warn'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              actionBackend === 'playwright' ? 'bg-ok'
              : actionBackend === 'chrome_mcp' ? 'bg-info'
              : 'bg-warn'
            }`}></span>
            Actions: {actionBackend === 'playwright' ? 'Playwright Real'
              : actionBackend === 'chrome_mcp' ? 'Chrome MCP Real'
              : 'Mocked'}
          </span>

          {/* Chrome MCP — tool-call transport availability only; unrelated to
              whether browser actions are real (see Actions badge above). */}
          <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium ${
            runtimeStatus.chrome_mcp_status === 'connected'
              ? 'bg-ok/10 border-ok/30 text-ok'
              : 'bg-surface-hover border-line text-ink-muted'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              runtimeStatus.chrome_mcp_status === 'connected' ? 'bg-ok' : 'bg-warn'
            }`}></span>
            Chrome MCP: {
              runtimeStatus.chrome_mcp_status === 'connected' ? 'Connected'
              : runtimeStatus.chrome_mcp_status === 'not_configured' ? 'Not configured'
              : 'Unavailable'
            }
          </span>
        </div>
      )}
        </div>
      </details>

      {/* ── Observation warning (downgrade notice) ────────────────────────── */}
      {obsWarning && (
        <div className="flex items-start gap-3 bg-warn/10 border border-warn/30 rounded-xl px-5 py-3 text-xs text-warn">
          <span className="text-warn flex-shrink-0">{Icons.Alert}</span>
          {obsWarning}
        </div>
      )}

      {/* ── Incoming session notice ───────────────────────────────────────── */}
      {incomingSession && (
        <div className="flex items-center gap-3 bg-info/10 border border-info/30 rounded-xl px-5 py-3 text-xs">
          <span className="w-2 h-2 rounded-full bg-info flex-shrink-0"></span>
          <span className="text-info">
            Launched from Chat recommendation
            {incomingSession.recommended_reason ? ` — ${incomingSession.recommended_reason}` : ''}.
          </span>
          {incomingSession.source_message_id && (
            <span className="ml-auto text-info font-mono">msg:{incomingSession.source_message_id.slice(0, 8)}</span>
          )}
        </div>
      )}

      {/* ── Goal entry ───────────────────────────────────────────────────── */}
      <div className="bg-surface-sunken border border-line rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-ink">Browser Agent</h2>
          <span className={`text-xs font-mono border rounded px-2 py-0.5 ${
            browserBackend === 'automatic' ? 'bg-accent/10 text-accent border-accent/30'
            : obsActive && realActionsActive ? 'bg-ok/10 text-ok border-ok/30'
            : obsActive ? 'bg-accent/10 text-accent border-accent/30'
            : 'bg-surface-hover text-ink-muted border-line-strong'
          }`}>
            {browserBackend === 'automatic' ? 'Automatic'
              : obsActive && realActionsActive ? 'Live observation · Real low-risk actions'
              : obsActive ? 'Live observation · Mock actions'
              : 'Mock mode'}
          </span>
        </div>
        <p className="text-xs text-ink-faint">
          Theriad chooses the available isolated browser automatically for live web tasks.
          Ordinary current-information questions use automatic web search; sensitive actions
          still pause for approval.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-ink-muted mb-1">Goal</label>
            <textarea
              value={goal}
              onChange={e => setGoal(e.target.value)}
              placeholder="Find the cheapest LLM API for summarization tasks."
              rows={2}
              disabled={hasSession && !isTerminal}
              className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:border-accent disabled:opacity-50 resize-none"
            />
          </div>

          <details className="rounded-lg border border-line bg-surface-hover px-3 py-2">
            <summary className="cursor-pointer text-xs font-medium text-ink-muted">
              Advanced options
            </summary>
            <div className="pt-3 space-y-3">
              <label className="block text-xs text-ink-muted mb-1">Live Browser Backend</label>
              <select
                value={browserBackend}
                onChange={e => {
                  const value = e.target.value;
                  setBrowserBackend(value);
                  setRealObsEnabled(value !== 'automatic' && value !== 'mock');
                  setRealActionEnabled(value !== 'automatic' && value !== 'mock');
                }}
                disabled={hasSession && !isTerminal}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent disabled:opacity-50"
              >
                <option value="automatic">Automatic (recommended)</option>
                {(caps?.available_backends || []).filter(name => name !== 'extension').map(name => <option key={name} value={name}>{name}</option>)}
              </select>
              {browserBackend === 'automatic' && (
                <p className="text-[11px] text-ink-faint">Theriad selects an available isolated backend at task start.</p>
              )}
              <label className="block text-xs text-ink-muted mb-1">Start URL override</label>
              <input
                value={startUrl}
                onChange={e => setStartUrl(e.target.value)}
                placeholder="https://site.com"
                disabled={hasSession && !isTerminal}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:border-accent disabled:opacity-50"
              />
              <p className="mt-1 text-[11px] text-ink-faint">
                Usually unnecessary—Theriad infers a site or web search from the goal.
              </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-ink-muted mb-1">Max Steps</label>
              <input
                type="number"
                value={maxSteps}
                onChange={e => setMaxSteps(parseInt(e.target.value) || 20)}
                min={1} max={100}
                disabled={hasSession && !isTerminal}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-xs text-ink-muted mb-1">Permission Mode</label>
              <select
                value={permissionMode}
                onChange={e => setPermissionMode(e.target.value)}
                disabled={hasSession && !isTerminal}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent disabled:opacity-50"
              >
                <option value="always_ask">Always Ask for Permission</option>
                <option value="ask_high_risk">Only Ask if Risk is High</option>
                <option value="never_ask">Never Ask for Permission</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-ink-muted mb-1">Max Cost USD (optional)</label>
              <input
                type="number"
                value={maxCost}
                onChange={e => setMaxCost(e.target.value)}
                placeholder="0.01"
                step="0.001"
                disabled={hasSession && !isTerminal}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:border-accent disabled:opacity-50"
              />
            </div>
          </div>

          {/* ── Real observation toggle ─────────────────────────────────────── */}
          <div className={`flex items-center justify-between rounded-lg px-4 py-3 border ${
            playwrightAvailable
              ? 'bg-surface-hover border-line'
              : 'bg-surface-sunken border-line opacity-60'
          }`}>
            <div>
              <p className="text-xs font-medium text-ink">Live Observation (Playwright)</p>
              <p className="text-xs text-ink-faint mt-0.5">
                {playwrightAvailable
                  ? 'Observe a live webpage with a headless browser.'
                  : caps?.install_hint
                  ? <span>Requires Chromium binaries — run: <code className="font-mono text-warn">{caps.install_hint}</code></span>
                  : 'Requires Playwright — not installed.'}
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={realObsEnabled}
              onClick={() => playwrightAvailable && setRealObsEnabled(v => !v)}
              disabled={!playwrightAvailable || (hasSession && !isTerminal)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                realObsEnabled && playwrightAvailable
                  ? 'bg-accent'
                  : 'bg-surface-hover'
              } disabled:cursor-not-allowed`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                realObsEnabled && playwrightAvailable ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </button>
          </div>

          {/* ── Real low-risk actions toggle ────────────────────────────────── */}
          <div className={`flex items-center justify-between rounded-lg px-4 py-3 border ${
            realActionsSupported
              ? 'bg-surface-hover border-line'
              : 'bg-surface-sunken border-line opacity-60'
          }`}>
            <div>
              <p className="text-xs font-medium text-ink">Real Low-Risk Actions (Playwright)</p>
              <p className="text-xs text-ink-faint mt-0.5">
                {realActionsSupported
                  ? 'Allow navigation, scrolling, and ordinary link clicks to run for real. Sensitive actions still require approval.'
                  : 'Requires Playwright + Chromium — not installed on the backend.'}
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={realActionEnabled}
              onClick={() => realActionsSupported && setRealActionEnabled(v => !v)}
              disabled={!realActionsSupported || (hasSession && !isTerminal)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                realActionEnabled && realActionsSupported
                  ? 'bg-ok'
                  : 'bg-surface-hover'
              } disabled:cursor-not-allowed`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                realActionEnabled && realActionsSupported ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </button>
          </div>
            </div>
          </details>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-2 pt-1">
          {(!hasSession || isTerminal) && (
            <button
              onClick={handleStart}
              disabled={loading || !goal.trim()}
              className={`px-4 py-2 disabled:opacity-40 text-ink text-sm font-medium rounded-lg transition-colors ${
                realActionEnabled && realActionsSupported ? 'bg-ok hover:bg-ok/85'
                : realObsEnabled && playwrightAvailable ? 'bg-accent hover:bg-accent-strong'
                : 'bg-surface-hover hover:bg-line-strong'
              }`}
            >
              {loading && !hasSession ? 'Starting…' : 'Start Agent'}
            </button>
          )}

          {/* Normal UX has no "Run N steps" controls (Part 2) — the agent
              runs continuously once started. Stop is the only control while
              running; Approve/Deny/Stop appear in the pending-approval card
              below while waiting_for_approval. */}
          {hasSession && !isTerminal && (
            <button
              onClick={handleStop}
              disabled={loading}
              className="px-4 py-2 bg-danger hover:bg-danger/85 disabled:opacity-40 text-on-accent text-sm font-medium rounded-lg transition-colors"
            >
              Stop
            </button>
          )}

          {isTerminal && hasSession && (
            <button
              onClick={() => { setSessionId(null); setStatus(null); setError(null); setObsWarning(null); }}
              className="px-4 py-2 bg-surface-hover hover:bg-line-strong text-ink text-sm font-medium rounded-lg transition-colors"
            >
              New Session
            </button>
          )}
        </div>

        {error && <ApiError msg={error} />}
      </div>

      {/* ── Session stats bar ─────────────────────────────────────────────── */}
      <LocalBrowserPanel agentSessionId={realObsActive ? sessionId : null} />

      {status && (
        <div className="bg-surface-sunken border border-line rounded-xl p-4">
          <div className="flex flex-wrap gap-4 text-xs">
            <div>
              <span className="text-ink-faint">Status</span>
              <span className={`ml-2 font-semibold ${statusColor}`}>
                {isWaiting ? 'Waiting for permission'
                  : isWaitingForInput ? 'Waiting for input'
                  : isCaptchaPause ? 'CAPTCHA needs attention'
                  : status.status?.replace(/_/g, ' ')}
              </span>
            </div>
            <div>
              <span className="text-ink-faint">Steps</span>
              <span className="ml-2 text-ink font-mono">{status.steps_taken ?? 0}</span>
            </div>
            <div>
              <span className="text-ink-faint">Model calls</span>
              <span className="ml-2 text-ink font-mono">{status.model_calls ?? 0}</span>
            </div>
            <div>
              <span className="text-ink-faint">Cost</span>
              <span className="ml-2 text-ok font-mono">${(status.estimated_cost_usd ?? 0).toFixed(6)}</span>
            </div>
            <div>
              <span className="text-ink-faint">Elapsed</span>
              <span className="ml-2 text-ink font-mono">{(status.elapsed_seconds ?? 0).toFixed(1)}s</span>
            </div>
            {status.current_url && (
              <div className="min-w-0">
                <span className="text-ink-faint">URL</span>
                <span className="ml-2 text-accent font-mono truncate max-w-xs inline-block align-bottom">{status.current_url}</span>
              </div>
            )}
          </div>
          {/* Surface backend error message when the session fails */}
          {status.status === 'failed' && status.error && (
            <div className="mt-3 flex items-start gap-2 text-xs text-danger bg-danger/10 border border-danger/30 rounded-lg px-3 py-2">
              <span className="flex-shrink-0">{Icons.Alert}</span>
              <span className="font-mono break-all">{status.error}</span>
            </div>
          )}
          {status.loop_stop_reason && status.loop_stop_reason !== 'no_progress' && (
            <p className="mt-2 text-xs text-ink-faint">Stop reason: <span className="font-mono">{status.loop_stop_reason}</span></p>
          )}
        </div>
      )}

      {isCaptchaPause && (
        <div className="bg-warn/10 border border-warn/40 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2">
            {Icons.Alert}
            <span className="text-warn font-semibold text-sm">CAPTCHA encountered</span>
          </div>
          <p className="text-sm text-ink">
            {status.browser_handoff_ready
              ? 'Theriad opened a temporary browser window. Complete the CAPTCHA there, then continue.'
              : 'Theriad tried automatic protection handling but could not clear this challenge.'}
          </p>
          {status.page_title && (
            <p className="text-xs text-ink-muted">Page: {status.page_title}</p>
          )}
          {status.current_url && (
            <p className="text-xs font-mono text-ink-faint break-all">{status.current_url}</p>
          )}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handleCaptchaHandoff('/api/agent/browser/continue_captcha')}
              disabled={loading}
              className="px-3 py-2 text-xs rounded-lg border border-ok/30 text-ok bg-ok/10 hover:bg-ok/20"
            >
              {status.browser_handoff_ready ? 'I finished it — continue' : 'Try again'}
            </button>
            <button
              onClick={() => handleCaptchaHandoff('/api/agent/browser/use_api_search')}
              disabled={loading}
              className="px-3 py-2 text-xs rounded-lg border border-accent/30 text-accent bg-accent/10 hover:bg-accent/20"
            >
              Use API search instead
            </button>
            <button
              onClick={handleStop}
              disabled={loading}
              className="px-3 py-2 text-xs rounded-lg border border-danger/30 text-danger bg-danger/10 hover:bg-danger/20"
            >
              Stop task
            </button>
          </div>
        </div>
      )}

      {/* ── Pending approval card ─────────────────────────────────────────── */}
      {status?.pending_approval && (
        <div className="bg-warn/10 border border-warn/40 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2">
            {Icons.Alert}
            <span className="text-warn font-semibold text-sm">
              {status.pending_approval.required_input ? 'Credentials required' : 'Permission required'}
            </span>
            <span className={`text-xs font-mono ml-auto ${AGENT_RISK_COLOR[status.pending_approval.risk_level] ?? 'text-ink-muted'}`}>
              risk: {status.pending_approval.risk_level}
            </span>
          </div>
          <p className="text-sm text-ink">
            Theriad wants to {describeAgentAction(status.pending_approval.action_type)}
            {status.pending_approval.target ? <> on <span className="font-mono text-warn">{status.pending_approval.target}</span></> : null}.
          </p>
          {/* D-045: when the planner needs input this release cannot capture
              (currently: login credentials), say so plainly instead of
              implying that Approve alone will make it succeed. */}
          {status.pending_approval.required_input && (
            <p className="text-sm text-warn font-medium">
              Credentials are required to continue. Approval alone is not enough.
            </p>
          )}
          <p className="text-xs text-ink-muted">Reason: {status.pending_approval.reason}</p>
          <div className="flex flex-wrap gap-4 text-xs text-ink-faint">
            {status.pending_approval.confidence != null && (
              <span>Confidence: <span className="text-ink font-mono">{(status.pending_approval.confidence * 100).toFixed(0)}%</span></span>
            )}
            {status.pending_approval.current_url && (
              <span className="truncate max-w-xs">Page: <span className="text-ink font-mono">{status.pending_approval.current_url}</span></span>
            )}
            {status.pending_approval.input_fields?.length > 0 && (
              <span>Fields needed: <span className="text-ink font-mono">{status.pending_approval.input_fields.join(', ')}</span></span>
            )}
          </div>
          {status.pending_approval.data_to_submit && (
            <div className="text-xs bg-surface-sunken border border-line rounded-lg p-2">
              <p className="text-ink-faint mb-1 uppercase tracking-wide">Data to be submitted</p>
              <pre className="font-mono text-ink whitespace-pre-wrap">{JSON.stringify(status.pending_approval.data_to_submit, null, 2)}</pre>
            </div>
          )}
          <div className="flex gap-2 pt-1">
            {status.pending_approval.required_input ? (
              <>
                <button
                  onClick={handleApprove}
                  disabled={loading}
                  className="px-4 py-2 bg-ok hover:bg-ok/85 disabled:opacity-40 text-on-accent text-sm font-medium rounded-lg transition-colors"
                >
                  Provide credentials, later
                </button>
                <button
                  onClick={handleDeny}
                  disabled={loading}
                  className="px-4 py-2 bg-danger hover:bg-danger/85 disabled:opacity-40 text-on-accent text-sm font-medium rounded-lg transition-colors"
                >
                  Skip login / find another path
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={handleApprove}
                  disabled={loading}
                  className="px-4 py-2 bg-ok hover:bg-ok/85 disabled:opacity-40 text-on-accent text-sm font-medium rounded-lg transition-colors"
                >
                  Approve
                </button>
                <button
                  onClick={handleDeny}
                  disabled={loading}
                  className="px-4 py-2 bg-danger hover:bg-danger/85 disabled:opacity-40 text-on-accent text-sm font-medium rounded-lg transition-colors"
                >
                  Deny
                </button>
              </>
            )}
            <button
              onClick={handleStop}
              disabled={loading}
              className="px-4 py-2 bg-surface-hover hover:bg-line-strong disabled:opacity-40 text-ink text-sm font-medium rounded-lg transition-colors"
            >
              Stop
            </button>
          </div>
        </div>
      )}

      {/* ── Waiting-for-input card (D-045) ────────────────────────────────────
          Shown once the user has already responded to a credentials-required
          approval and the session honestly stopped rather than pretending to
          continue. pending_approval is cleared by this point — status alone
          drives this card. */}
      {!status?.pending_approval && isWaitingForInput && (
        <div className="bg-warn/10 border border-warn/40 rounded-xl p-5 space-y-2">
          <div className="flex items-center gap-2">
            {Icons.Alert}
            <span className="text-warn font-semibold text-sm">Credentials required to continue</span>
          </div>
          <p className="text-sm text-ink">
            {status?.error || 'Credentials required to continue.'}
          </p>
          <p className="text-xs text-ink-muted">
            Credential capture isn't implemented in this release, so Theriad stopped here
            rather than guessing. Start a new session once you're able to provide credentials
            another way, or use Stop to end this session.
          </p>
          <div className="flex gap-2 pt-1">
            <button
              onClick={handleStop}
              disabled={loading}
              className="px-4 py-2 bg-surface-hover hover:bg-line-strong disabled:opacity-40 text-ink text-sm font-medium rounded-lg transition-colors"
            >
              Stop
            </button>
          </div>
        </div>
      )}

      {/* ── Last observation ──────────────────────────────────────────────── */}
      {status?.last_observation && (
        <div className="bg-surface-sunken border border-line rounded-xl p-4">
          <p className="text-xs text-ink-faint mb-2 font-semibold uppercase tracking-wide">Last Observation</p>
          <p className="text-xs text-ink font-mono whitespace-pre-wrap leading-relaxed">{status.last_observation}</p>
        </div>
      )}

      {/* ── Final result ──────────────────────────────────────────────────── */}
      {status?.final_result && (
        <div className="bg-ok/10 border border-ok/30 rounded-xl p-5">
          <p className="text-xs text-ok mb-2 font-semibold uppercase tracking-wide">Final Result</p>
          <pre className="text-sm text-ink whitespace-pre-wrap font-mono leading-relaxed">{status.final_result}</pre>
        </div>
      )}

      {/* ── Action log ───────────────────────────────────────────────────── */}
      {status?.action_log?.length > 0 && (
        <div className="bg-surface-sunken border border-line rounded-xl p-4">
          <p className="text-xs text-ink-faint mb-3 font-semibold uppercase tracking-wide">Action Log</p>
          <div className="space-y-1.5">
            {[...status.action_log].reverse().map((entry, i) => (
              <div key={i} className="flex items-start gap-3 text-xs font-mono">
                <span className="text-ink-faint w-6 text-right shrink-0">#{entry.step}</span>
                <span className={`shrink-0 w-16 ${AGENT_RESULT_COLOR[entry.result] ?? 'text-ink-muted'}`}>{entry.result}</span>
                <span className="text-accent shrink-0">{entry.action_type}</span>
                {entry.target && <span className="text-ink-muted truncate">{entry.target}</span>}
                {entry.summary && <span className="text-ink-faint truncate ml-auto">{entry.summary}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Step diagnostics ─────────────────────────────────────────────────
           Collapsed by default (Part 7): internal step diagnostics are kept
           for debugging/auditability but are not the primary UX — the
           status bar, action log, and pending-approval card above are. ──── */}
      {status?.step_diagnostics?.length > 0 && (
        <details className="bg-surface-sunken border border-line rounded-xl p-4">
          <summary className="text-xs text-ink-faint font-semibold uppercase tracking-wide cursor-pointer select-none">
            Step Diagnostics ({status.step_diagnostics.length})
          </summary>
          <div className="overflow-x-auto mt-3">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-ink-faint border-b border-line">
                  <th className="text-left py-1 pr-3">#</th>
                  <th className="text-left py-1 pr-3">Subtask</th>
                  <th className="text-left py-1 pr-3">Tier</th>
                  <th className="text-left py-1 pr-3">Mode</th>
                  <th className="text-left py-1 pr-3">Approval</th>
                  <th className="text-left py-1 pr-3">Latency</th>
                  <th className="text-left py-1">Note</th>
                </tr>
              </thead>
              <tbody>
                {status.step_diagnostics.map((d, i) => (
                  <tr key={i} className="border-b border-line hover:bg-surface-hover">
                    <td className="py-1 pr-3 text-ink-faint">{d.step_id}</td>
                    <td className="py-1 pr-3 text-accent">{d.subtask_type?.replace('agent_', '')}</td>
                    <td className="py-1 pr-3 text-ink">{d.tier_used ?? '—'}</td>
                    <td className="py-1 pr-3">
                      {d.deterministic
                        ? <span className="text-ok">deterministic</span>
                        : <span className="text-warn">model</span>}
                    </td>
                    <td className="py-1 pr-3 text-ink-muted">
                      {d.approval_state && d.approval_state !== 'none' ? d.approval_state : '—'}
                    </td>
                    <td className="py-1 pr-3 text-ink-muted">{d.latency_ms ? `${d.latency_ms}ms` : '—'}</td>
                    <td className="py-1 text-ink-faint truncate max-w-xs">{d.reason ?? d.error ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {/* ── Status footer ────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between text-xs text-ink-faint pb-2 border-t border-line pt-3">
        <span>
          {obsActive && realActionsActive
            ? `Live observation + real low-risk actions (${actionBackend}). Sensitive actions require approval.`
            : obsActive
            ? `Live observation (${obsBackend}) — the selected backend reads the page. Actions are mocked.`
            : 'Mock mode — no real browser launched. Observation and actions are both mocked.'}
        </span>
        <span className="font-mono">
          obs: {obsBackend} · actions: {actionBackend}
        </span>
      </div>
    </div>
  );
}

// The auth route is intentionally shown over the same workspace surface users
// see after sign-in. Keep it inert and visually quiet: the form remains the
// only interactive surface while the blurred /app view gives the sign-in state
// real Theriad context.
function AuthBackdrop() {
  return (
    <div className="theriad-auth-backdrop" aria-hidden="true" inert>
      <div className="theriad-auth-backdrop__site">
        <TheriadDashboard preview />
      </div>
      <div className="theriad-auth-backdrop__pattern">
        <span className="theriad-auth-backdrop__glow" />
      </div>
      <div className="theriad-auth-backdrop__veil" />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth modal — create an account (migrates guest history) or sign in
// ─────────────────────────────────────────────────────────────────────────────
function AuthModal({ reason, mfa, initialTab, onClose, onAuthed, onContinueAsGuest, required = false }) {
  const [tab, setTab] = useState(() => initialTab || (reason === 'initial' ? 'login' : 'register')); // 'register' | 'login'
  const [view, setView] = useState('auth'); // 'auth' | 'recovery'
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [recoveryIdentifier, setRecoveryIdentifier] = useState('');
  const [recoveryKey, setRecoveryKey] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [verificationSent, setVerificationSent] = useState(false);
  const [recoveryNotice, setRecoveryNotice] = useState('');
  const [mfaState, setMfaState] = useState(mfa);
  const [mfaCode, setMfaCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [guestBusy, setGuestBusy] = useState(false);
  const [recoveryBusy, setRecoveryBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setMfaState(mfa);
    if (initialTab === 'login' || initialTab === 'register') setTab(initialTab);
    else if (reason === 'initial') setTab('login');
  }, [initialTab, mfa, reason]);

  useEffect(() => {
    const query = window.location.hash.split('?')[1] || '';
    const recoveryKeyFromEmail = new URLSearchParams(query).get('recovery_key');
    if (recoveryKeyFromEmail) {
      setView('recovery');
      setVerificationSent(true);
      setRecoveryKey(recoveryKeyFromEmail.toUpperCase());
    }
  }, []);

  async function close() {
    if (required) return;
    if (mfa) await cancelSupabaseSignIn().catch(() => {});
    onClose?.();
  }

  async function handleGoogle() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await signInWithGoogle();
    } catch (err) {
      setError(toFriendlyError(err.message));
      setBusy(false);
    }
  }

  function openRecovery() {
    setView('recovery');
    setRecoveryIdentifier('');
    setRecoveryKey('');
    setNewPassword('');
    setVerificationSent(false);
    setRecoveryNotice('');
    setError(null);
  }

  function returnToAuth() {
    setView('auth');
    setTab('login');
    setError(null);
    setRecoveryNotice('');
  }

  async function submit(e) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      if (mfa) {
        const user = await verifySupabaseEmailCode(mfaState.challengeId, mfaCode.trim());
        onAuthed(user);
        return;
      }
      if (tab === 'register') {
        onAuthed(await authRegister(username.trim(), password, email.trim()));
        return;
      }
      onAuthed(await authLogin(username.trim(), password));
    } catch (err) {
      setError(toFriendlyError(err.message));
    } finally {
      setBusy(false);
    }
  }

  async function continueAsGuest() {
    if (busy || guestBusy || !onContinueAsGuest) return;
    setGuestBusy(true);
    setError(null);
    try {
      onAuthed(await onContinueAsGuest());
    } catch (err) {
      setError(toFriendlyError(err?.message || err));
    } finally {
      setGuestBusy(false);
    }
  }

  async function submitRecovery(e) {
    e.preventDefault();
    if (recoveryBusy) return;
    setRecoveryBusy(true);
    setError(null);
    try {
      if (!verificationSent) {
        const result = await requestPasswordRecovery(recoveryIdentifier.trim());
        setRecoveryNotice(result.message || 'Check your inbox for a verification key.');
        setVerificationSent(true);
      } else {
        onAuthed(await completePasswordRecovery(recoveryKey.trim(), newPassword));
      }
    } catch (err) {
      setError(toFriendlyError(err.message));
    } finally {
      setRecoveryBusy(false);
    }
  }

  async function resendMfaCode() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const next = await resendSupabaseEmailCode();
      setMfaState(next);
      setMfaCode('');
    } catch (err) {
      setError(toFriendlyError(err.message));
    } finally {
      setBusy(false);
    }
  }

  if (mfa) {
    return (
      <div className="theriad-auth-modal-scrim fixed inset-0 z-[90] grid place-items-center p-5">
        <div className="w-full max-w-sm rounded-2xl border border-line-strong bg-surface-raised p-6 shadow-2xl">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-[0.18em] text-accent">Google sign-in</p>
              <h2 className="mt-2 text-lg font-semibold text-ink">Check your email</h2>
              <p className="mt-1 text-xs text-ink-muted leading-relaxed">
                We sent a six-digit verification code to {mfaState?.email || 'your email address'}. Enter it here to finish signing in.
              </p>
            </div>
            {!required && (
              <button onClick={close} aria-label="Cancel sign-in" className="text-ink-faint hover:text-ink transition-colors">✕</button>
            )}
          </div>

          <form onSubmit={submit} className="mt-5 space-y-4">
            <div role="status" aria-live="polite" className="rounded-lg border border-ok/30 bg-ok/10 px-3 py-3">
              <p className="text-xs font-semibold text-ink">Email sent</p>
              <p className="mt-1 text-[11px] leading-relaxed text-ink-muted">
                The code expires in {Math.max(1, Math.ceil((mfaState?.expiresIn || 600) / 60))} minutes and can be used once.
              </p>
            </div>
            <label className="block">
              <span className="text-xs text-ink-muted">Email verification code</span>
              <input
                value={mfaCode}
                onChange={event => setMfaCode(event.target.value.replace(/\D/g, '').slice(0, 6))}
                inputMode="numeric"
                autoComplete="one-time-code"
                pattern="[0-9]{6}"
                minLength={6}
                maxLength={6}
                autoFocus
                required
                className="mt-1 w-full rounded-lg border border-line bg-surface-sunken px-3 py-3 text-center text-lg tracking-[0.45em] text-ink outline-none focus:border-accent/60"
              />
            </label>
            {error && <p className="text-xs text-danger bg-danger/10 border border-danger/30 rounded-lg px-3 py-2">{error}</p>}
            <button
              type="submit"
              disabled={busy || mfaCode.length !== 6}
              className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent hover:bg-accent-strong disabled:opacity-50 transition-colors"
            >
              {busy ? 'Verifying…' : 'Verify and continue'}
            </button>
            <button
              type="button"
              onClick={resendMfaCode}
              disabled={busy}
              className="w-full text-center text-xs text-ink-muted hover:text-ink disabled:opacity-50 transition-colors"
            >
              {busy ? 'Sending…' : 'Resend code'}
            </button>
          </form>
          {!error && <p className="mt-4 text-center text-[11px] text-ink-faint">If you don’t see it, check your spam folder.</p>}
        </div>
      </div>
    );
  }

  if (view === 'recovery') {
    return (
      <div className="theriad-auth-modal-scrim fixed inset-0 z-[90] grid place-items-center p-5">
        <div className="w-full max-w-sm max-h-[calc(100dvh-1rem)] overflow-y-auto rounded-2xl border border-line-strong bg-surface-raised p-5 shadow-2xl">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-[0.18em] text-accent">Account recovery</p>
              <h2 className="mt-2 text-lg font-semibold text-ink">Recover your account</h2>
              <p className="mt-1 text-xs leading-relaxed text-ink-muted">
                Enter your username or account email. We will email a one-time verification key if that account has a recovery email.
              </p>
            </div>
            {!required && (
              <button onClick={close} aria-label="Close" className="min-h-11 min-w-11 text-ink-faint hover:text-ink transition-colors">✕</button>
            )}
          </div>

          <form onSubmit={submitRecovery} className="mt-4 space-y-3">
            {!verificationSent ? (
              <label className="block">
                <span className="text-xs text-ink-muted">Username or account email</span>
                <input
                  value={recoveryIdentifier}
                  onChange={event => setRecoveryIdentifier(event.target.value)}
                  autoComplete="username"
                  minLength={3}
                  required
                  className="mt-1 w-full rounded-lg border border-line bg-surface-sunken px-3 py-3 text-sm text-ink outline-none focus:border-accent/60"
                />
              </label>
            ) : (
              <>
                <div role="status" aria-live="polite" className="rounded-lg border border-ok/30 bg-ok/10 px-3 py-3">
                  <p className="text-xs font-semibold text-ink">Check your email</p>
                  <p className="mt-1 text-[11px] leading-relaxed text-ink-muted">
                    {recoveryNotice || 'Enter the verification key from your email and choose a new password.'}
                  </p>
                </div>
                <label className="block">
                  <span className="text-xs text-ink-muted">Verification key</span>
                  <input
                    value={recoveryKey}
                    onChange={event => setRecoveryKey(event.target.value.toUpperCase())}
                    placeholder="THR-ABCD-1234"
                    autoComplete="one-time-code"
                    spellCheck="false"
                    required
                    className="mt-1 w-full rounded-lg border border-line bg-surface-sunken px-3 py-3 font-mono text-sm tracking-[0.05em] text-ink outline-none focus:border-accent/60"
                  />
                </label>
                <label className="block">
                  <span className="text-xs text-ink-muted">New password</span>
                  <input
                    type="password"
                    value={newPassword}
                    onChange={event => setNewPassword(event.target.value)}
                    autoComplete="new-password"
                    minLength={6}
                    required
                    className="mt-1 w-full rounded-lg border border-line bg-surface-sunken px-3 py-3 text-sm text-ink outline-none focus:border-accent/60"
                  />
                </label>
              </>
            )}
            {error && <p className="text-xs text-danger bg-danger/10 border border-danger/30 rounded-lg px-3 py-2">{error}</p>}
            <button
              type="submit"
              disabled={recoveryBusy || (!verificationSent && recoveryIdentifier.trim().length < 3) || (verificationSent && (!recoveryKey.trim() || newPassword.length < 6))}
              className="min-h-11 w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent hover:bg-accent-strong disabled:opacity-50 transition-colors"
            >
              {recoveryBusy ? 'One moment…' : verificationSent ? 'Reset password' : 'Email verification key'}
            </button>
            {verificationSent && (
              <button
                type="button"
                onClick={() => {
                  setVerificationSent(false);
                  setRecoveryKey('');
                  setNewPassword('');
                  setError(null);
                  setRecoveryNotice('');
                }}
                className="min-h-11 w-full text-center text-xs text-ink-muted hover:text-ink transition-colors"
              >
                Use a different username or email
              </button>
            )}
          </form>

          <div className="mt-4 rounded-lg border border-line bg-surface-sunken/60 px-3 py-3">
            <p className="text-xs font-semibold text-ink">Signed in with Google?</p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-muted">
              Google manages that account’s password. Use Google sign-in or Google’s own recovery flow to restore access.
            </p>
            <button
              type="button"
              onClick={handleGoogle}
              disabled={busy || !isSupabaseConfigured()}
              className="mt-3 inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-lg border border-line-strong bg-surface-raised px-3 py-2 text-xs font-semibold text-ink hover:border-accent/50 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
            >
              <GoogleMark />
              <span>{busy ? 'Opening Google…' : 'Continue with Google'}</span>
            </button>
            {!isSupabaseConfigured() && <p className="mt-2 text-[10px] text-ink-faint">Google sign-in is not configured in this environment.</p>}
          </div>

          <button
            type="button"
            onClick={returnToAuth}
            className="mt-2 min-h-11 w-full text-center text-xs text-ink-muted hover:text-ink transition-colors"
          >
            Back to sign in
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="theriad-auth-modal-scrim fixed inset-0 z-[90] overflow-y-auto">
      <div className="theriad-auth-modal-stage">
        <section className="theriad-auth-card" aria-labelledby="auth-title">
          <div className="theriad-auth-form-panel">
            <div className="theriad-auth-brand" aria-label="Theriad">
              <TheriadLogo
                className="theriad-auth-brand__mark"
                imageClassName="theriad-auth-brand__mark-image"
              />
              <span>Theriad</span>
            </div>

            <header className="theriad-auth-header">
              <div>
                <p className="theriad-auth-kicker">Account access</p>
                <h2 id="auth-title" className="theriad-auth-title">
                  {tab === 'register' ? 'Create your account' : reason === 'initial' ? 'Welcome to Theriad' : 'Welcome back'}
                </h2>
                <p className="theriad-auth-copy">
                  {reason === 'initial'
                    ? tab === 'login'
                      ? 'Sign in to keep your chats and projects tied to your account.'
                      : 'Create a free account to keep your chats and projects tied to you.'
                    : tab === 'register'
                    ? 'Create an account to keep your chats and projects private to you.'
                    : 'Sign in to get back to your own history.'}
                </p>
              </div>
              {!required && (
                <button type="button" onClick={close} aria-label="Close sign in" className="theriad-auth-close">✕</button>
              )}
            </header>

            <form onSubmit={submit} className="theriad-auth-form">
              <div className="theriad-auth-fields">
                <label className="theriad-auth-field">
                  <span className="theriad-auth-field__label">Username</span>
                  <input
                    value={username}
                    onChange={e => setUsername(e.target.value)}
                    autoComplete="username"
                    minLength={3}
                    required
                    className="theriad-auth-input"
                  />
                </label>
                {tab === 'register' && (
                  <label className="theriad-auth-field">
                    <span className="theriad-auth-field__label">Recovery email</span>
                    <input
                      type="email"
                      value={email}
                      onChange={e => setEmail(e.target.value)}
                      autoComplete="email"
                      required
                      className="theriad-auth-input"
                    />
                    <span className="theriad-auth-field__hint">Used only to help you recover your account.</span>
                  </label>
                )}
                <label className="theriad-auth-field">
                  <span className="theriad-auth-field__label">Password</span>
                  <input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    autoComplete={tab === 'register' ? 'new-password' : 'current-password'}
                    minLength={6}
                    required
                    className="theriad-auth-input"
                  />
                </label>
              </div>

              {error && (
                <p role="alert" className="theriad-auth-error">{error}</p>
              )}

              <button type="submit" disabled={busy} className="theriad-auth-primary">
                <span>{busy ? 'One moment…' : tab === 'register' ? 'Create account' : 'Sign in'}</span>
              </button>
            </form>

            <div className="theriad-auth-links">
              {tab === 'login' && (
                <button type="button" onClick={openRecovery} className="theriad-auth-link">Forgot password?</button>
              )}
              <button
                type="button"
                onClick={() => { setTab(t => (t === 'register' ? 'login' : 'register')); setError(null); }}
                className="theriad-auth-link theriad-auth-link--muted"
              >
                {tab === 'register' ? 'Already have an account? Sign in' : 'New here? Create an account'}
              </button>
            </div>

            {tab === 'register' && (
              <p className="theriad-auth-note">
                No credit card, no charge on sign-up. You start on the free tier and only pay if you add credits later.
              </p>
            )}

            <div className="theriad-auth-more">
              <div className="theriad-auth-divider" aria-hidden="true">
                <span>or continue with</span>
              </div>
              <button
                type="button"
                onClick={handleGoogle}
                disabled={busy || !isSupabaseConfigured()}
                className="theriad-auth-google-button"
                title={!isSupabaseConfigured() ? 'Add the Supabase frontend variables to enable Google sign-in.' : undefined}
              >
                <GoogleMark />
                <span>{busy ? 'Opening Google…' : 'Continue with Google'}</span>
              </button>
              {onContinueAsGuest && (
                <button
                  type="button"
                  onClick={continueAsGuest}
                  disabled={busy || guestBusy}
                  className="theriad-auth-guest-button"
                >
                  {guestBusy ? 'Starting guest mode…' : 'Continue as guest'}
                </button>
              )}
              {!isSupabaseConfigured() && (
                <p className="theriad-auth-more__hint">Google sign-in becomes available after Supabase is configured.</p>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function TheriadDashboard({ initialView = 'Chat', preview = false }) {
  const [currentView, setCurrentView]     = useState(initialView);
  const [isPending, startTransition]      = useTransition();
  const showRouteProgress = useDelayedLoading(isPending);
  const [openProject, setOpenProject]     = useState(null);
  // Chat nav state: hold a conversation id to open on the Chat page
  const [chatConvId, setChatConvId]       = useState(null);
  const [blankChat, setBlankChat]         = useState(true);
  const [newChatRequest, setNewChatRequest] = useState(0);
  // Mobile sidebar drawer state
  const [sidebarOpen, setSidebarOpen]     = useState(false);
  // Session: a visitor chooses sign-in or guest mode in the auth modal. A
  // Google session is not considered ready until the server has verified the
  // email code sent after Google sign-in.
  const [user, setUser]                   = useState(() => (preview ? AUTH_BACKDROP_USER : getCachedUser()));
  const [authReady, setAuthReady]         = useState(() => preview || Boolean(getCachedUser()));
  const [authFailure, setAuthFailure]     = useState(null);
  const [authModal, setAuthModal]         = useState(null); // null | { type: 'manual'|'mfa', ... }

  // The drawer dims the page behind it, so Escape should dismiss it the way it
  // dismisses the other overlays in the workspace.
  useEffect(() => {
    if (!sidebarOpen) return undefined;
    const onKeyDown = event => { if (event.key === 'Escape') setSidebarOpen(false); };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [sidebarOpen]);

  const openAuth = useCallback((initialTab = 'register') => {
    setAuthFailure(null);
    setAuthModal({
      type: 'manual',
      reason: 'manual',
      initialTab: initialTab === 'login' ? 'login' : 'register',
    });
  }, []);

  useEffect(() => {
    if (preview) return undefined;

    let cancelled = false;

    const onUserChange = e => {
      if (cancelled) return;
          setUser(e.detail);
          if (e.detail) setAuthFailure(null);
        };
    const onStorageChange = event => {
      if (cancelled || (event.key && !event.key.includes('theriad_auth_user'))) return;
      const nextUser = getCachedUser();
      setUser(nextUser);
      if (!nextUser) setAuthModal({ type: 'manual', reason: 'initial' });
    };
    const onOpenAuth   = event => openAuth(event?.detail?.initialTab);
    window.addEventListener('theriad:user-changed', onUserChange);
    window.addEventListener('storage', onStorageChange);
    window.addEventListener('theriad:open-auth', onOpenAuth);

    // Restore an existing account or create the free guest workspace. The
    // first useful screen should be available immediately; account creation
    // remains the path for preserving the workspace and enabling paid use.
    ensureSession({ createGuest: true })
      .then(restoredUser => {
        if (cancelled) return;
        setUser(restoredUser);
        if (!restoredUser) setAuthModal({ type: 'manual', reason: 'initial' });
      })
      .catch(error => {
        if (cancelled) return;
        if (error?.code === 'SUPABASE_EMAIL_MFA_REQUIRED') {
          setAuthModal({ type: 'mfa', ...(error.details || {}) });
        } else {
          setAuthFailure(error?.message || 'Unable to restore your Theriad session.');
        }
      })
      .finally(() => { if (!cancelled) setAuthReady(true); });

    return () => {
      cancelled = true;
      window.removeEventListener('theriad:user-changed', onUserChange);
      window.removeEventListener('storage', onStorageChange);
      window.removeEventListener('theriad:open-auth', onOpenAuth);
    };
  }, [openAuth, preview]);

  async function continueAsGuest() {
    const guest = await ensureSession({ createGuest: true });
    if (!guest?.is_guest) throw new Error('Unable to start guest mode.');
    return guest;
  }

  function handleNavChange(view) {
    // Route/Simulate/Analyze are legacy view names. Route details now live
    // inline in Chat, so stale callers should land there instead.
    const nextView = ['Route', 'Simulate', 'Analyze'].includes(view) ? 'Chat' : view;
    startTransition(() => {
      setCurrentView(nextView);
      if (nextView !== 'Projects') setOpenProject(null);
      if (nextView !== 'Chat') {
        setChatConvId(null);
        setBlankChat(false);
      } else {
        setBlankChat(false);
      }
    });
  }

  function handleNewChat() {
    startTransition(() => {
      setCurrentView('Chat');
      setOpenProject(null);
      setChatConvId(null);
      setBlankChat(true);
      setNewChatRequest(value => value + 1);
    });
  }

  /**
   * Navigate from Recents to the appropriate view.
   * conversation → Chat page at that conversation
   * project      → Projects → ProjectDetail
   * file/run     → Projects for now (detail pages are future work)
   */
  function handleRecentsNavigate(item) {
    if (item.type === 'conversation') {
      setChatConvId(item.id);
      setCurrentView('Chat');
    } else if (item.type === 'project') {
      setOpenProject({ id: item.project_id, name: item.project_name ?? item.title });
      setCurrentView('Projects');
    } else {
      // fall through to Projects for files/runs
      setCurrentView('Projects');
    }
  }

  const headerTitle = openProject
    ? openProject.name
    : currentView;

  const renderView = () => {
    if (currentView === 'Chat') {
      return (
        <ChatPage
          key={chatConvId ?? 'new'}
          initialConvId={chatConvId}
          blankChat={blankChat}
          newChatRequest={newChatRequest}
          preview={preview}
          isGuest={Boolean(user?.is_guest)}
          onOpenAuth={openAuth}
        />
      );
    }
    if (currentView === 'Recents') {
      return <RecentsPage onNavigate={handleRecentsNavigate} />;
    }
    if (currentView === 'Projects') {
      if (openProject) {
        return (
          <ProjectDetail
            project={openProject}
            onBack={() => setOpenProject(null)}
          />
        );
      }
      return (
        <ProjectsScreen onOpen={p => { setOpenProject(p); }} />
      );
    }
    switch (currentView) {
      case 'Analyze':  return <AnalyzeScreen />;
      case 'Models':   return <ModelsScreen />;
      case 'Costs':    return <CostsScreen />;
      case 'Logs':     return <LogsScreen />;
      case 'Analytics': return <SiteAnalyticsScreen user={user} />;
      case 'Skills':   return <SkillsScreen />;
      case 'Memory':   return <MemoryScreen />;
      case 'Settings': return <SettingsScreen user={user} />;
      default:         return <AnalyzeScreen />;
    }
  };

  if (!authReady && !user) {
    return <LoadingScreen
      variant="workspace"
      title="Restoring your Theriad session"
      message="Checking identity and security status…"
    />;
  }

  // A cached guest remains available while Google OAuth completes so free
  // workspace state is not discarded. The email-code challenge must still be
  // a hard auth gate: render it before the guest shell instead of depending on
  // `user` being empty or on a second in-shell modal.
  if (authModal?.type === 'mfa') {
    return (
      <>
        <AuthBackdrop />
        <AuthModal
          mfa={authModal}
          onClose={() => {
            setAuthModal(null);
            setAuthFailure('Google sign-in was canceled. Sign in to continue.');
          }}
          onAuthed={signedInUser => { setUser(signedInUser); setAuthFailure(null); setAuthModal(null); }}
        />
      </>
    );
  }

  if (!user && authModal?.type === 'manual') {
    return (
      <>
        <AuthBackdrop />
        <AuthModal
          reason={authModal.reason}
          initialTab={authModal.initialTab}
          required={authModal.reason === 'initial'}
          onClose={() => setAuthModal(null)}
          onContinueAsGuest={authModal.reason === 'initial' ? continueAsGuest : undefined}
          onAuthed={signedInUser => { setUser(signedInUser); setAuthFailure(null); setAuthModal(null); }}
        />
      </>
    );
  }

  if (!user && authFailure) {
    return (
      <div className="grid h-dvh place-items-center bg-surface px-5 text-ink">
        <div className="w-full max-w-md rounded-2xl border border-line-strong bg-surface-raised p-6 shadow-xl">
          <p className="text-[10px] uppercase tracking-[0.18em] text-danger">Sign-in setup issue</p>
          <h1 className="mt-2 text-lg font-semibold">Theriad could not restore your session</h1>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">{authFailure}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-5 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent hover:bg-accent-strong transition-colors"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    // Shell background intentionally transparent: body carries var(--surface)
    // plus the aurora ambient glow (body::before); opaque panels sit above both.
    // h-screen + overflow-hidden let the Chat flex layout fill the viewport exactly.
    <div className="theriad-app-shell h-dvh overflow-hidden text-ink flex font-sans antialiased">
      <style>{`
        body { font-family: 'Manrope Variable', var(--font-sans); }
        .font-mono { font-family: var(--font-mono); }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }
        @keyframes spin { to { transform: rotate(360deg); } }
        .animate-spin { animation: spin 1s linear infinite; }
      `}</style>

      <div aria-hidden="true" className="aurora-topline" />

      {authFailure && (
        <div className="pointer-events-none fixed inset-x-0 top-3 z-[100] flex justify-center px-4">
          <div
            role="alert"
            className="pointer-events-auto flex w-full max-w-xl items-start justify-between gap-4 rounded-xl border border-danger/30 bg-surface-raised px-4 py-3 shadow-2xl"
          >
            <div className="min-w-0">
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-danger">Sign-in needs attention</p>
              <p className="mt-1 text-xs leading-relaxed text-ink-muted">{authFailure}</p>
            </div>
            <div className="flex flex-shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={() => openAuth('login')}
                className="rounded-lg border border-accent/30 px-3 py-1.5 text-xs font-semibold text-accent hover:bg-accent/10"
              >
                Try again
              </button>
              <button
                type="button"
                aria-label="Dismiss sign-in error"
                onClick={() => setAuthFailure(null)}
                className="px-1 text-ink-faint hover:text-ink"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      )}

      {authModal?.type === 'manual' && (
        <AuthModal
          reason={authModal.reason}
          initialTab={authModal.initialTab}
          onClose={() => setAuthModal(null)}
          onAuthed={u => { setUser(u); setAuthFailure(null); setAuthModal(null); }}
        />
      )}

      {/* ── Mobile/tablet top bar ── visible below the desktop rail breakpoint */}
      <div className="theriad-mobile-bar lg:hidden fixed top-0 left-0 right-0 h-14 bg-surface border-b border-line z-40 flex items-center px-4 gap-3 flex-shrink-0">
        <button
          onClick={() => setSidebarOpen(true)}
          aria-label="Open navigation"
          className="p-2 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-hover transition-colors flex-shrink-0"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6"/>
            <line x1="3" y1="12" x2="21" y2="12"/>
            <line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        <span className="theriad-mobile-title text-sm font-semibold text-ink flex-1 truncate">
          <span className="theriad-mobile-mark" aria-hidden="true">T</span>
          <span className="truncate">{headerTitle}</span>
        </span>
      </div>

      {/* ── Mobile overlay — dims content when sidebar is open */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <Sidebar
        current={currentView}
        setCurrent={handleNavChange}
        onNewChat={handleNewChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        user={user}
        preview={preview}
      />

      {/* min-w-0 prevents the flex child from blowing out past the viewport.
          overflow-x-hidden is a second safety net for any stragglers.
          pt-14 on mobile accounts for the fixed top bar height.
          For Chat: overflow-hidden keeps the inner flex column from leaking.
          For other views: overflow-y-auto enables page-level scrolling. */}
      <main className={`theriad-main ${currentView === 'Chat' ? 'theriad-main--chat' : 'theriad-main--scrollable'} flex-1 lg:ml-20 min-w-0 overflow-x-hidden ${
        currentView === 'Chat'
          ? 'h-full overflow-hidden p-2 pt-16 lg:p-4 lg:pt-4'
          : 'h-full overflow-y-auto p-4 pt-16 lg:p-8 lg:pt-8'
      }`} aria-busy={isPending}>
        {showRouteProgress && (
          <div className="theriad-route-progress" role="status" aria-live="polite">
            <span className="theriad-route-progress__dot" aria-hidden="true" />
            Loading {headerTitle}…
          </div>
        )}
        {currentView !== 'Chat' && (
          <header className="theriad-page-header mb-5 md:mb-8">
            <h1 className="text-xl md:text-2xl font-bold text-ink tracking-tight">{headerTitle}</h1>
          </header>
        )}
        <AppErrorBoundary
          key={`${currentView}:${openProject?.id ?? ''}`}
          resetKey={`${currentView}:${openProject?.id ?? ''}`}
        >
          {renderView()}
        </AppErrorBoundary>
      </main>
    </div>
  );
}
