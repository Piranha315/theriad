import { useEffect, useMemo, useState } from 'react';
import { DelayedLoader } from './components/LoadingState.jsx';

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value}s`;
  const minutes = Math.floor(value / 60);
  const remainder = value % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function formatDay(value) {
  const date = new Date(`${value}T12:00:00Z`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function Metric({ label, value, detail }) {
  return (
    <div className="rounded-2xl border border-line bg-surface-raised p-5">
      <p className="text-xs uppercase tracking-[0.16em] text-ink-faint">{label}</p>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-ink">{value}</p>
      {detail && <p className="mt-1 text-xs text-ink-muted">{detail}</p>}
    </div>
  );
}

function AnalyticsTable({ title, rows, nameKey, valueKey, valueLabel }) {
  return (
    <section className="rounded-2xl border border-line bg-surface-raised p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        <span className="text-[11px] uppercase tracking-wider text-ink-faint">{valueLabel}</span>
      </div>
      {rows.length === 0 ? (
        <p className="mt-6 text-sm text-ink-faint">No data yet.</p>
      ) : (
        <div className="mt-4 space-y-3">
          {rows.map(row => (
            <div key={row[nameKey]} className="flex items-center justify-between gap-4 text-sm">
              <span className="truncate text-ink-muted" title={row[nameKey]}>{row[nameKey]}</span>
              <span className="flex-shrink-0 font-mono text-ink">{Number(row[valueKey]).toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function SiteAnalyticsScreen({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [errorStatus, setErrorStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [excluded, setExcluded] = useState(null);
  const [exclusionLoading, setExclusionLoading] = useState(false);
  const [exclusionError, setExclusionError] = useState(null);
  const [passcode, setPasscode] = useState('');
  const [unlockLoading, setUnlockLoading] = useState(false);
  const [unlockError, setUnlockError] = useState(null);

  async function load() {
    setLoading(true);
    setError(null);
    setErrorStatus(null);
    try {
      const response = await fetch('/api/analytics/summary?days=30');
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        setErrorStatus(response.status);
        const detail = typeof body.detail === 'string' ? body.detail : `Analytics request failed (${response.status}).`;
        throw new Error(detail);
      }
      setData(body);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to load site analytics.');
    } finally {
      setLoading(false);
    }
  }

  async function unlockAnalytics(event) {
    event.preventDefault();
    if (!passcode.trim()) return;
    setUnlockLoading(true);
    setUnlockError(null);
    try {
      const response = await fetch('/api/analytics/unlock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ passcode }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof body.detail === 'string' ? body.detail : `Unlock request failed (${response.status}).`);
      }
      setPasscode('');
      await load();
      await loadExclusion();
    } catch (reason) {
      setUnlockError(reason instanceof Error ? reason.message : 'Unable to unlock site analytics.');
    } finally {
      setUnlockLoading(false);
    }
  }

  async function loadExclusion() {
    try {
      const response = await fetch('/api/analytics/exclusion');
      const body = await response.json().catch(() => ({}));
      if (!response.ok) return;
      setExcluded(Boolean(body.excluded));
    } catch {
      // The summary remains useful if the optional owner-browser control fails.
    }
  }

  async function updateExclusion() {
    if (excluded === null) return;
    setExclusionLoading(true);
    setExclusionError(null);
    try {
      const response = await fetch('/api/analytics/exclusion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !excluded }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}).`);
      }
      setExcluded(Boolean(body.excluded));
    } catch (reason) {
      setExclusionError(reason instanceof Error ? reason.message : 'Unable to update owner exclusion.');
    } finally {
      setExclusionLoading(false);
    }
  }

  useEffect(() => {
    load();
    loadExclusion();
    const handleUserChange = event => {
      // A fresh owner sign-in can happen from the sidebar while this private
      // route is open. Retry after the auth token has been stored.
      if (event.detail && !event.detail.is_guest) {
        load();
        loadExclusion();
      }
    };
    window.addEventListener('theriad:user-changed', handleUserChange);
    return () => window.removeEventListener('theriad:user-changed', handleUserChange);
  }, []);

  const peak = useMemo(
    () => Math.max(1, ...(data?.daily || []).map(day => Number(day.visits) || 0)),
    [data?.daily],
  );

  if (loading && !data) {
    return <DelayedLoader loading={loading} label="Loading site analytics…" className="rounded-2xl border border-line bg-surface-raised p-6" />;
  }

  if (error && !data) {
    return (
      <div className="rounded-2xl border border-warn/30 bg-surface-raised p-6">
        <h2 className="text-sm font-semibold text-ink">Site analytics unavailable</h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink-muted">{error}</p>
        {errorStatus === 403 && !user?.is_guest && (
          <form onSubmit={unlockAnalytics} className="mt-5 max-w-md">
            <label htmlFor="analytics-passcode" className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-faint">
              Analytics access code
            </label>
            <div className="mt-2 flex flex-wrap gap-2">
              <input
                id="analytics-passcode"
                type="password"
                value={passcode}
                onChange={event => setPasscode(event.target.value)}
                autoComplete="one-time-code"
                placeholder="Enter access code"
                className="min-w-[220px] flex-1 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                disabled={unlockLoading}
              />
              <button
                type="submit"
                disabled={unlockLoading || !passcode.trim()}
                className="rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-on-accent hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
              >
                {unlockLoading ? 'Unlocking…' : 'Unlock analytics'}
              </button>
            </div>
            <p className="mt-2 text-xs text-ink-faint">Use a registered Theriad account. Access lasts for 7 days on this account.</p>
            {unlockError && <p className="mt-2 text-xs text-warn">{unlockError}</p>}
          </form>
        )}
        <button
          type="button"
          onClick={load}
          className="mt-4 rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-on-accent hover:bg-accent-strong"
        >
          Try again
        </button>
      </div>
    );
  }

  const bounceRate = `${Math.round((Number(data.bounce_rate) || 0) * 100)}%`;
  return (
    <div className="space-y-5 pb-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm text-ink-muted">Private owner view · {data.window_days}-day activity window</p>
          <p className="mt-1 text-xs text-ink-faint">Updated {new Date(data.generated_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</p>
        </div>
        <button
          type="button"
          onClick={load}
          className="rounded-lg border border-line px-3 py-2 text-xs font-medium text-ink-muted hover:border-line-strong hover:text-ink"
        >
          Refresh
        </button>
      </div>

      {error && <p className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">{error}</p>}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <Metric label="Unique visitors" value={Number(data.total_visitors).toLocaleString()} detail="All time" />
        <Metric label="Visits" value={Number(data.total_visits).toLocaleString()} detail="All time sessions" />
        <Metric label="Page views" value={Number(data.total_page_views).toLocaleString()} detail="All tracked views" />
        <Metric label="Avg. stay" value={formatDuration(data.average_session_seconds)} detail="Activity-based estimate" />
        <Metric label="Active now" value={Number(data.active_now).toLocaleString()} detail="Seen in the last 2 minutes" />
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(280px,1fr)]">
        <section className="rounded-2xl border border-line bg-surface-raised p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-ink">Visits by day</h2>
              <p className="mt-1 text-xs text-ink-faint">A visit is a browser session with at least one page view.</p>
            </div>
            <span className="text-xs text-ink-muted">Bounce {bounceRate}</span>
          </div>
          <div className="mt-6 flex h-48 items-end gap-1.5 sm:gap-2">
            {(data.daily || []).map(day => (
              <div key={day.day} className="group flex min-w-0 flex-1 flex-col items-center justify-end gap-2">
                <span className="invisible text-[10px] text-ink-muted group-hover:visible">{day.visits}</span>
                <div
                  className="w-full max-w-8 rounded-t bg-accent/70 transition-colors group-hover:bg-accent"
                  style={{ height: `${Math.max(4, (Number(day.visits) / peak) * 130)}px` }}
                  title={`${day.visits} visits · ${day.page_views} page views`}
                />
                <span className="w-8 truncate text-center text-[9px] text-ink-faint">{formatDay(day.day)}</span>
              </div>
            ))}
          </div>
        </section>

        <AnalyticsTable title="Top pages" rows={data.top_pages || []} nameKey="path" valueKey="views" valueLabel="Views" />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <AnalyticsTable title="Referrers" rows={data.top_referrers || []} nameKey="referrer" valueKey="visits" valueLabel="Visits" />
        <section className="rounded-2xl border border-line bg-surface-raised p-5">
          <h2 className="text-sm font-semibold text-ink">Measurement notes</h2>
          <p className="mt-3 text-sm leading-relaxed text-ink-muted">
            Visitors are anonymous first-party cookie IDs. A session ends after 30 minutes of inactivity; duration comes from visible page activity and 30-second heartbeats.
          </p>
          <p className="mt-3 text-xs leading-relaxed text-ink-faint">No IP addresses, user agents, message contents, or account-linked identities are stored.</p>
          <div className="mt-5 border-t border-line pt-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-faint">Owner exclusion</p>
                <p className="mt-1 max-w-md text-xs leading-relaxed text-ink-muted">
                  Exclude this browser from future traffic counts. This is browser-based, not an IP filter, and does not remove historical visits.
                </p>
              </div>
              <button
                type="button"
                onClick={updateExclusion}
                disabled={exclusionLoading || excluded === null}
                className="rounded-lg border border-line px-3 py-2 text-xs font-medium text-ink-muted hover:border-line-strong hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
              >
                {exclusionLoading ? 'Saving…' : excluded ? 'Include this browser' : 'Exclude my visits'}
              </button>
            </div>
            {exclusionError && <p className="mt-2 text-xs text-warn">{exclusionError}</p>}
          </div>
        </section>
      </div>
    </div>
  );
}
