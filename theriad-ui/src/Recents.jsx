/**
 * Recents.jsx — Recent conversations, projects, files, and runs feed.
 */
import React, { useState, useEffect } from 'react';
import { toFriendlyError } from './uiErrors.js';
import { DelayedLoader } from './components/LoadingState.jsx';

async function apiFetch(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    ...rest,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} on GET ${path}: ${body}`);
  }
  return res.json();
}

function fmtDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso.slice(0, 16);
  }
}

function fmtCost(usd) {
  if (usd == null) return null;
  return `$${usd.toFixed(6)}`;
}

function fmtBytes(n) {
  if (n == null) return '';
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / 1024 / 1024).toFixed(1)}MB`;
}

const TYPE_ICON = {
  conversation: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" className="w-4 h-4">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  ),
  project: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" className="w-4 h-4">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
    </svg>
  ),
  file: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" className="w-4 h-4">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  ),
  run: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" className="w-4 h-4">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
  ),
};

const TYPE_COLOR = {
  conversation: 'text-accent',
  project:      'text-info',
  file:         'text-ink-muted',
  run:          'text-ok',
};

const TYPE_LABEL = {
  conversation: 'Conversation',
  project:      'Project',
  file:         'File',
  run:          'Run',
};

function TierBadge({ tier }) {
  if (!tier) return null;
  const cls = {
    free_or_floor:    'text-ok',
    cheap_bulk:       'text-ok',
    cheap_code:       'text-ok',
    code_strong:      'text-accent',
    strong_reasoning: 'text-accent',
    premium:          'text-warn',
  }[tier] ?? 'text-ink-muted';
  return <span className={`font-mono text-xs ${cls}`}>{tier}</span>;
}

function ItemCard({ item, onNavigate }) {
  const { type, title, project_name, updated_at, metadata } = item;

  function handleClick() {
    onNavigate(item);
  }

  return (
    <button
      onClick={handleClick}
      className="w-full text-left flex items-start gap-3 px-4 py-3 rounded-xl border border-line hover:border-line hover:bg-surface-hover transition-colors group"
    >
      <span className={`mt-0.5 flex-shrink-0 ${TYPE_COLOR[type]}`}>
        {TYPE_ICON[type]}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-ink text-sm font-medium truncate">{title}</span>
          <span className="text-[10px] text-ink-faint flex-shrink-0 uppercase tracking-widest">
            {TYPE_LABEL[type]}
          </span>
        </div>
        {project_name && type !== 'project' && (
          <p className="text-xs text-ink-faint truncate mt-0.5">in {project_name}</p>
        )}
        {type === 'conversation' && item.snippet && (
          <p className="text-xs text-ink-muted/80 truncate mt-1">{item.snippet}</p>
        )}
        {/* Metadata row */}
        <div className="flex items-center gap-3 mt-1">
          {type === 'run' && metadata && (
            <>
              <TierBadge tier={metadata.tier} />
              {metadata.actual_cost_usd != null && (
                <span className="text-xs font-mono text-ok">{fmtCost(metadata.actual_cost_usd)}</span>
              )}
              {metadata.total_latency_ms != null && (
                <span className="text-xs text-ink-faint">{metadata.total_latency_ms?.toFixed(0)}ms</span>
              )}
              {metadata.status && (
                <span className={`text-xs ${metadata.status === 'success' ? 'text-ok' : 'text-warn'}`}>
                  {metadata.status}
                </span>
              )}
            </>
          )}
          {type === 'file' && metadata && (
            <>
              <span className="text-xs text-ink-faint">{fmtBytes(metadata.size_bytes)}</span>
              {metadata.mime_type && (
                <span className="text-xs text-ink-faint">{metadata.mime_type}</span>
              )}
            </>
          )}
        </div>
      </div>
      <span className="text-[10px] text-ink-faint flex-shrink-0 mt-0.5">{fmtDate(updated_at)}</span>
    </button>
  );
}

const FILTER_OPTIONS = ['all', 'conversation', 'project', 'file', 'run'];

// ─────────────────────────────────────────────────────────────────────────────
// Chat search helper
// Scores items by relevance to a query string.
// Returns items sorted by score (desc), then by updated_at (desc).
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {object[]} items  - array of recent items
 * @param {string}   query  - raw search string from the user
 * @returns {object[]}      - filtered and sorted items
 */
function searchChats(items, query) {
  const q = query.trim().toLowerCase();
  if (!q) return items;

  function score(item) {
    const title   = (item.title ?? '').toLowerCase();
    const project = (item.project_name ?? '').toLowerCase();
    const date    = (item.updated_at ?? '').toLowerCase();
    // Snippet: first message or metadata summary — fall back to empty
    const snippet = (item.snippet ?? item.first_message ?? '').toLowerCase();

    let s = 0;
    if (title === q)             s += 100;
    else if (title.startsWith(q)) s +=  60;
    else if (title.includes(q))  s +=  40;
    if (snippet.includes(q))     s +=  20;
    if (project.includes(q))     s +=  10;
    if (date.includes(q))        s +=   5;
    return s;
  }

  return items
    .map(item => ({ item, s: score(item) }))
    .filter(({ s }) => s > 0)
    .sort((a, b) => {
      if (b.s !== a.s) return b.s - a.s;
      // Secondary sort: most recently updated first
      const ta = new Date(a.item.updated_at ?? 0).getTime();
      const tb = new Date(b.item.updated_at ?? 0).getTime();
      return tb - ta;
    })
    .map(({ item }) => item);
}

export default function RecentsPage({ onNavigate }) {
  const [items, setItems]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [filter, setFilter]     = useState('all');
  const [search, setSearch]     = useState('');

  useEffect(() => {
    setLoading(true);
    apiFetch('/api/recents?limit=50')
      .then(data => { setItems(data.items ?? []); setLoading(false); })
      .catch(e => { setError(toFriendlyError(e)); setLoading(false); });
  }, []);

  // Apply type filter first, then search filter
  const typeFiltered = filter === 'all' ? items : items.filter(i => i.type === filter);
  const visible = search.trim() ? searchChats(typeFiltered, search) : typeFiltered;

  const isSearchActive = search.trim().length > 0;

  function handleNavigate(item) {
    if (onNavigate) onNavigate(item);
  }

  return (
    <div>
      {/* Search bar */}
      <div className="relative mb-4">
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint pointer-events-none">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
        </div>
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search chats…"
          className="w-full bg-surface-raised border border-line rounded-xl py-2.5 pl-10 pr-4 text-sm text-ink placeholder-ink-faint focus:outline-none focus:border-accent/50 transition-colors"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink transition-colors"
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-1 mb-6 flex-wrap">
        {FILTER_OPTIONS.map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-colors ${
              filter === f
                ? 'bg-accent/10 text-accent border border-accent/30'
                : 'text-ink-faint hover:text-ink border border-transparent'
            }`}
          >
            {f}
          </button>
        ))}
        <span className="ml-auto text-xs text-ink-faint">{visible.length} items</span>
      </div>

      {/* Content */}
      <DelayedLoader loading={loading} label="Loading recent activity…" className="py-8" />

      {error && (
        <div className="text-warn text-xs p-4 bg-warn/10 rounded-lg border border-warn/20">
          {error}
        </div>
      )}

      {!loading && !error && visible.length === 0 && (
        <div className="text-center py-12 text-ink-faint text-sm">
          {isSearchActive
            ? 'No chats found.'
            : `No recent ${filter === 'all' ? 'activity' : filter + 's'} yet.`}
        </div>
      )}

      <div className="space-y-2">
        {visible.map(item => (
          <ItemCard key={`${item.type}-${item.id}`} item={item} onNavigate={handleNavigate} />
        ))}
      </div>
    </div>
  );
}
