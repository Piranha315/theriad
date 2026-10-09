/**
 * Projects feature — project list, project detail with tabs:
 *   Conversation · Files · Runs · Context · Settings
 *
 * Uses the same design system as Theriad.jsx (dark palette, Tailwind CSS, inline SVG icons).
 * No additional npm dependencies.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import CostConfirmationDialog from './CostConfirmationDialog.jsx';
import { toFriendlyError } from './uiErrors.js';
import { DelayedLoader } from './components/LoadingState.jsx';

// ── API helper ────────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  // Build headers first, then spread remaining options so headers are never
  // silently overwritten by a spread of options that also contains a headers key.
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    ...rest,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    const method = (rest.method ?? 'GET').toUpperCase();
    const msg = `API ${res.status} on ${method} ${path}: ${body}`;
    console.error('[Projects] API error', { method, path, status: res.status, body });
    throw new Error(msg);
  }
  return res.json();
}

// Variant for requests where a non-2xx response without JSON body is possible
// (e.g. DELETE 204 No Content — caller should use plain fetch for those).
async function apiDelete(path) {
  const res = await fetch(path, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    const msg = `API ${res.status} on DELETE ${path}: ${body}`;
    console.error('[Projects] DELETE error', { path, status: res.status, body });
    throw new Error(msg);
  }
  // 204 No Content — no body to parse
}

// ── Shared primitives ─────────────────────────────────────────────────────────
const Icon = ({ path, className = 'w-4 h-4' }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
       className={className}>{path}</svg>
);

const Icons = {
  Plus:    <Icon path={<><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></>} />,
  Folder:  <Icon path={<><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></>} />,
  Trash:   <Icon path={<><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></>} />,
  Upload:  <Icon path={<><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></>} />,
  File:    <Icon path={<><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></>} />,
  Send:    <Icon path={<><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></>} />,
  Back:    <Icon path={<><polyline points="15 18 9 12 15 6"/></>} />,
  Refresh: <Icon path={<><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></>} />,
  Alert:   <Icon path={<><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></>} />,
  Chat:    <Icon path={<><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></>} />,
  Chip:    <Icon path={<><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M20 15h2M2 9h2M20 9h2M9 2v2M9 20v2"/></>} />,
  Check:   <Icon path={<><polyline points="20 6 9 17 4 12"/></>} />,
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

const ApiError = ({ msg }) => (
  <div className="text-warn text-xs p-3 bg-warn/10 rounded-lg border border-warn/20 whitespace-pre-wrap">
    {toFriendlyError(msg)}
  </div>
);

const Badge = ({ children, color = 'slate' }) => {
  const colors = {
    cyan:    'bg-accent/10 text-accent border-accent/30',
    emerald: 'bg-ok/10 text-ok border-ok/30',
    amber:   'bg-warn/10 text-warn border-warn/30',
    slate:   'bg-ink-faint/10 text-ink-muted border-line-strong',
    red:     'bg-danger/10 text-danger border-danger/30',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${colors[color] ?? colors.slate}`}>
      {children}
    </span>
  );
};

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

// ── Projects List Screen ──────────────────────────────────────────────────────
export function ProjectsScreen({ onOpen }) {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName]   = useState('');
  const [newDesc, setNewDesc]   = useState('');

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    apiFetch('/api/projects')
      .then(setProjects)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleCreate(e) {
    e.preventDefault();
    if (!newName.trim()) return;
    try {
      // Use the returned project directly — no second GET needed.
      const project = await apiFetch('/api/projects', {
        method: 'POST',
        body: JSON.stringify({ name: newName.trim(), description: newDesc.trim() || null }),
      });
      setNewName('');
      setNewDesc('');
      setCreating(false);
      setProjects(prev => [project, ...prev]);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleDelete(id, e) {
    e.stopPropagation();
    if (!confirm('Delete this project? This cannot be undone.')) return;
    try {
      await apiDelete(`/api/projects/${id}`);
      setProjects(prev => prev.filter(p => p.id !== id));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="max-w-4xl mx-auto pb-12">
      <div className="flex flex-col items-start justify-between gap-3 mb-6 sm:flex-row sm:items-center">
        <p className="text-ink-muted text-sm">Organize prompts, conversations, and files into projects.</p>
        <button
          onClick={() => setCreating(true)}
          className="flex items-center space-x-2 bg-accent/20 hover:bg-accent/30 text-accent px-4 py-2 rounded-lg text-sm font-medium border border-accent/30 transition-colors"
        >
          {Icons.Plus}
          <span>New Project</span>
        </button>
      </div>

      {error && <div className="mb-4"><ApiError msg={error} /></div>}

      {creating && (
        <form onSubmit={handleCreate} className="mb-6 bg-surface-raised border border-accent/30 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-ink mb-4">New Project</h3>
          <div className="grid grid-cols-1 gap-4 mb-4 sm:grid-cols-2">
            <div>
              <label className="block text-xs text-ink-muted mb-1">Name *</label>
              <input
                autoFocus
                value={newName}
                onChange={e => setNewName(e.target.value)}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink outline-none focus:border-accent/50"
                placeholder="My Refactor Project"
              />
            </div>
            <div>
              <label className="block text-xs text-ink-muted mb-1">Description</label>
              <input
                value={newDesc}
                onChange={e => setNewDesc(e.target.value)}
                className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink outline-none focus:border-accent/50"
                placeholder="Optional description"
              />
            </div>
          </div>
          <div className="flex justify-end space-x-2">
            <button type="button" onClick={() => setCreating(false)} className="text-ink-muted hover:text-ink text-sm px-4 py-2">Cancel</button>
            <button type="submit" className="bg-accent/20 hover:bg-accent/30 text-accent px-4 py-2 rounded-lg text-sm font-medium border border-accent/30">Create</button>
          </div>
        </form>
      )}

      {loading ? <DelayedLoader loading={loading} label="Loading projects…" className="py-4" /> : (
        projects.length === 0 ? (
          <div className="text-center py-16 text-ink-faint text-sm">
            <div className="w-12 h-12 mx-auto rounded-full bg-surface-hover flex items-center justify-center text-ink-faint mb-4">
              {React.cloneElement(Icons.Folder, { key: 'f' })}
            </div>
            <p>No projects yet. Create one to get started.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {projects.map(p => (
              <div
                key={p.id}
                onClick={() => onOpen(p)}
                className="bg-surface-raised border border-line hover:border-accent/40 rounded-xl p-5 cursor-pointer transition-colors flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between group"
              >
                <div className="flex items-center space-x-4 min-w-0">
                  <div className="text-ink-faint group-hover:text-accent transition-colors flex-shrink-0">
                    {Icons.Folder}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-ink truncate">{p.name}</p>
                    {p.description && (
                      <p className="text-xs text-ink-faint truncate mt-0.5">{p.description}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center space-x-3 flex-shrink-0">
                  <span className="text-xs text-ink-faint">{fmtDate(p.updated_at).split(',')[0]}</span>
                  <button
                    onClick={e => handleDelete(p.id, e)}
                    className="text-ink-faint hover:text-danger transition-colors opacity-0 group-hover:opacity-100"
                  >
                    {Icons.Trash}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )
      )}
    </div>
  );
}

// ── Project Detail ────────────────────────────────────────────────────────────
export function ProjectDetail({ project: initialProject, onBack }) {
  const [project, setProject]         = useState(initialProject);
  const [tab, setTab]                 = useState('conversation');
  const [conversations, setConvs]     = useState([]);
  const [activeConv, setActiveConv]   = useState(null);
  const [messages, setMessages]       = useState([]);
  const [files, setFiles]             = useState([]);
  const [runs, setRuns]               = useState([]);
  const [attachedFiles, setAttached]  = useState([]);  // file ids attached to current prompt
  const [prompt, setPrompt]           = useState('');
  const policy = 'balanced';
  // Keep the wire fields at their existing defaults while routing stays automatic.
  const speedPreference = 50;
  const qualityPreference = 50;
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [pendingCost, setPendingCost] = useState(null);
  const [instrDraft, setInstrDraft]   = useState(project.instructions || '');
  const [savedInstr, setSavedInstr]   = useState(false);
  const [ctxData, setCtxData]         = useState(null);
  const [ctxLoading, setCtxLoading]   = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef   = useRef(null);

  const load = useCallback(async () => {
    const [convRows, fileRows, runRows] = await Promise.all([
      apiFetch(`/api/projects/${project.id}/conversations`).catch(() => []),
      apiFetch(`/api/projects/${project.id}/files`).catch(() => []),
      apiFetch(`/api/projects/${project.id}/runs`).catch(() => []),
    ]);
    setConvs(convRows);
    setFiles(fileRows);
    setRuns(runRows);
    if (!activeConv && convRows.length > 0) {
      loadConversation(convRows[0].id);
    }
  }, [project.id]);  // eslint-disable-line

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function loadConversation(convId) {
    const data = await apiFetch(`/api/conversations/${convId}`).catch(() => null);
    if (data) {
      setActiveConv(data);
      setMessages(data.messages || []);
    }
  }

  async function handleNewConversation() {
    const conv = await apiFetch(`/api/projects/${project.id}/conversations`, {
      method: 'POST',
      body: JSON.stringify({ title: 'New Conversation' }),
    }).catch(e => { setError(e.message); return null; });
    if (conv) {
      setConvs(prev => [conv, ...prev]);
      setActiveConv(conv);
      setMessages([]);
    }
  }

  async function handleExecute(costDecision = null) {
    if (!prompt.trim() || !activeConv) return;
    setLoading(true);
    setError(null);
    try {
      if (!costDecision) {
        const preview = await apiFetch('/api/chat/cost-preview', {
          method: 'POST',
          body: JSON.stringify({
            message: prompt,
            policy,
            speed_preference: speedPreference,
            quality_preference: qualityPreference,
            project_id: project.id,
            conversation_id: activeConv.id,
            file_ids: attachedFiles,
          }),
        });
        if (preview.requires_confirmation) {
          setPendingCost(preview);
          return;
        }
      }
      const execution = await apiFetch('/api/execute', {
        method: 'POST',
        body: JSON.stringify({
          prompt,
          policy,
          speed_preference: speedPreference,
          quality_preference: qualityPreference,
          allow_paid_calls: true,
          project_id: project.id,
          conversation_id: activeConv.id,
          file_ids: attachedFiles,
          cost_decision: costDecision,
        }),
      });
      if (execution.persistence?.status === 'failed') {
        throw new Error(
          execution.persistence.error ||
          'The provider completed, but the conversation could not be saved.'
        );
      }
      // Refresh conversation messages
      await loadConversation(activeConv.id);
      // Refresh runs
      const runRows = await apiFetch(`/api/projects/${project.id}/runs`).catch(() => runs);
      setRuns(runRows);
      setPrompt('');
      setAttached([]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    const form = new FormData();
    form.append('file', f);
    try {
      // File upload must use FormData — do NOT set Content-Type header
      // (browser sets it automatically with the correct boundary).
      const path = `/api/projects/${project.id}/files`;
      const result = await fetch(path, { method: 'POST', body: form });
      if (!result.ok) {
        const body = await result.text().catch(() => result.statusText);
        const msg = `API ${result.status} on POST ${path}: ${body}`;
        console.error('[Projects] file upload error', { path, status: result.status, body });
        throw new Error(msg);
      }
      const fileRow = await result.json();
      setFiles(prev => [fileRow, ...prev]);
    } catch (err) {
      setError(err.message);
    }
    e.target.value = '';
  }

  async function handleDeleteFile(fileId) {
    if (!confirm('Delete this file?')) return;
    try {
      await apiDelete(`/api/files/${fileId}`);
      setFiles(prev => prev.filter(f => f.id !== fileId));
      setAttached(prev => prev.filter(id => id !== fileId));
    } catch (err) {
      setError(err.message);
    }
  }

  function toggleAttach(fileId) {
    setAttached(prev =>
      prev.includes(fileId) ? prev.filter(id => id !== fileId) : [...prev, fileId]
    );
  }

  async function handleSaveInstructions() {
    const updated = await apiFetch(`/api/projects/${project.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ instructions: instrDraft }),
    }).catch(e => { setError(e.message); return null; });
    if (updated) {
      setProject(updated);
      setSavedInstr(true);
      setTimeout(() => setSavedInstr(false), 2000);
    }
  }

  async function handleBuildContext() {
    setCtxLoading(true);
    const data = await apiFetch('/api/context/build', {
      method: 'POST',
      body: JSON.stringify({
        project_id: project.id,
        conversation_id: activeConv?.id,
        file_ids: attachedFiles,
      }),
    }).catch(e => { setError(e.message); return null; });
    if (data) setCtxData(data);
    setCtxLoading(false);
  }

  const TABS = [
    { id: 'conversation', label: 'Conversation' },
    { id: 'files',        label: 'Files' },
    { id: 'runs',         label: `Runs (${runs.length})` },
    { id: 'context',      label: 'Context' },
    { id: 'settings',     label: 'Settings' },
  ];

  const estTokens = Math.max(1, Math.round(prompt.split(/\s+/).filter(Boolean).length * 1.33));

  return (
    <div className="theriad-project-detail max-w-5xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <button onClick={onBack} aria-label="Back to projects" title="Back to projects" className="text-ink-muted hover:text-ink p-1 rounded">{Icons.Back}</button>
        <div className="flex min-w-0 items-center gap-3">
          <div className="text-ink-muted">{Icons.Folder}</div>
          <div className="min-w-0">
            <h2 className="break-words text-lg font-bold leading-tight text-ink">{project.name}</h2>
            {project.description && <p className="break-words text-xs text-ink-faint">{project.description}</p>}
          </div>
        </div>
      </div>

      {error && <div className="mb-4"><ApiError msg={error} /></div>}

      {/* Tabs */}
      <div className="theriad-project-tabs flex gap-1 overflow-x-auto border-b border-line mb-6">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`shrink-0 whitespace-nowrap px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              tab === t.id
                ? 'bg-surface-raised text-ink border-b-2 border-accent -mb-px'
                : 'text-ink-faint hover:text-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Conversation Tab ─────────────────────────────────────────────────── */}
      {tab === 'conversation' && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
          {/* Sidebar: conversation list */}
          <div className="lg:col-span-1">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-ink-faint uppercase tracking-wider">Threads</span>
              <button onClick={handleNewConversation} className="text-ink-muted hover:text-accent transition-colors">
                {Icons.Plus}
              </button>
            </div>
            <div className="space-y-1">
              {conversations.map(c => (
                <button
                  key={c.id}
                  onClick={() => loadConversation(c.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors truncate ${
                    activeConv?.id === c.id
                      ? 'bg-accent/10 text-accent'
                      : 'text-ink-muted hover:bg-surface-hover'
                  }`}
                >
                  {c.title || 'Untitled'}
                </button>
              ))}
              {conversations.length === 0 && (
                <p className="text-xs text-ink-faint italic px-2">No conversations yet</p>
              )}
            </div>
          </div>

          {/* Main: message history + composer */}
          <div className="lg:col-span-3 flex flex-col">
            {/* Messages */}
            <div className="flex-1 bg-surface-raised border border-line rounded-xl p-4 mb-4 min-h-[300px] max-h-[480px] overflow-y-auto space-y-4">
              {messages.length === 0 && (
                <div className="text-ink-faint text-sm text-center py-12">
                  {activeConv ? 'No messages yet. Start the conversation below.' : 'Select or create a conversation.'}
                </div>
              )}
              {messages.map(msg => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-xl px-4 py-3 text-sm ${
                    msg.role === 'user'
                      ? 'bg-accent/15 text-accent border border-accent/20'
                      : 'bg-surface text-ink border border-line'
                  }`}>
                    <p className={`text-xs mb-1 ${msg.role === 'user' ? 'text-accent' : 'text-ink-faint'}`}>
                      {msg.role === 'user' ? 'You' : 'Assistant'}
                    </p>
                    <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">{msg.content}</pre>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-surface border border-line rounded-xl px-4 py-3">
                    <Spinner label="Generating response…" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Attached files chips */}
            {attachedFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2">
                {attachedFiles.map(fid => {
                  const f = files.find(x => x.id === fid);
                  return (
                    <span key={fid} className="flex items-center space-x-1 px-2 py-1 bg-info/10 border border-info/30 rounded text-xs text-info">
                      <span>{f?.original_name ?? fid}</span>
                      <button onClick={() => toggleAttach(fid)} className="text-info hover:text-danger ml-1">✕</button>
                    </span>
                  );
                })}
                <span className="text-xs text-warn/70 flex items-center space-x-1">
                  {Icons.Alert}
                  <span>File content will be sent to provider</span>
                </span>
              </div>
            )}

            {/* Composer */}
            <div className="bg-surface-raised border border-line rounded-xl overflow-visible">
              <textarea
                className="w-full bg-transparent p-4 text-ink outline-none resize-none placeholder-ink-faint text-sm font-sans min-h-[80px]"
                placeholder={activeConv ? "Enter prompt…" : "Create or select a conversation first"}
                disabled={!activeConv}
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) setShowConfirm(true); }}
              />
              <div className="flex justify-between items-center p-3 border-t border-line bg-surface">
                <div className="flex items-center space-x-3">
                  <span className="text-xs text-ink-faint">~{estTokens} tokens</span>
                  {/* Attach file button */}
                  <div className="relative">
                    <button
                      className="text-xs text-ink-faint hover:text-ink flex items-center space-x-1"
                      onClick={() => setTab('files')}
                      title="Go to Files tab to attach"
                    >
                      {Icons.File}
                      <span>Attach file</span>
                    </button>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    disabled={!activeConv || !prompt.trim() || loading}
                    onClick={() => handleExecute()}
                    className="flex items-center space-x-1.5 bg-surface-hover hover:bg-line-strong disabled:opacity-40 text-ink px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                  >
                    {Icons.Send}
                    <span>Execute</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Files Tab ────────────────────────────────────────────────────────── */}
      {tab === 'files' && (
        <div>
          <div className="flex flex-col items-start gap-3 mb-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-ink-faint">Upload .txt .md .json .csv .py .js .jsx .ts .tsx .html .css files</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center space-x-2 bg-accent/20 hover:bg-accent/30 text-accent px-4 py-2 rounded-lg text-sm font-medium border border-accent/30"
            >
              {Icons.Upload}
              <span>Upload File</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".txt,.md,.json,.csv,.py,.js,.jsx,.ts,.tsx,.html,.css"
              onChange={handleFileUpload}
            />
          </div>

          {files.length === 0 ? (
            <div className="text-center py-16 text-ink-faint text-sm">No files uploaded yet.</div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-line">
              <table className="w-full min-w-[42rem] bg-surface-raised text-sm">
                <thead className="bg-surface text-ink-faint text-xs uppercase tracking-wider">
                  <tr>
                    <th className="text-left p-3 font-medium">Name</th>
                    <th className="text-left p-3 font-medium">Type</th>
                    <th className="text-right p-3 font-medium">Size</th>
                    <th className="text-left p-3 font-medium">Uploaded</th>
                    <th className="text-center p-3 font-medium">Attach</th>
                    <th className="text-center p-3 font-medium">Delete</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {files.map(f => (
                    <tr key={f.id} className="hover:bg-surface-hover">
                      <td className="p-3 font-mono text-ink text-xs">{f.original_name}</td>
                      <td className="p-3 text-ink-faint text-xs">{f.mime_type ?? '—'}</td>
                      <td className="p-3 text-right font-mono text-xs text-ink-muted">{fmtBytes(f.size_bytes)}</td>
                      <td className="p-3 text-xs text-ink-faint">{fmtDate(f.created_at).split(',')[0]}</td>
                      <td className="p-3 text-center">
                        <button
                          onClick={() => toggleAttach(f.id)}
                          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                            attachedFiles.includes(f.id)
                              ? 'bg-info/20 border border-info/40 text-info'
                              : 'bg-surface-hover text-ink-muted hover:bg-line-strong'
                          }`}
                        >
                          {attachedFiles.includes(f.id) ? 'Attached' : 'Attach'}
                        </button>
                      </td>
                      <td className="p-3 text-center">
                        <button
                          onClick={() => handleDeleteFile(f.id)}
                          className="text-ink-faint hover:text-danger transition-colors"
                        >
                          {Icons.Trash}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {attachedFiles.length > 0 && (
            <div className="mt-4 flex items-center space-x-2 text-xs text-warn bg-warn/10 border border-warn/20 rounded-lg p-3">
              {Icons.Alert}
              <span>{attachedFiles.length} file(s) attached — content will be sent to the AI provider when you execute.</span>
            </div>
          )}
        </div>
      )}

      {/* ── Runs Tab ─────────────────────────────────────────────────────────── */}
      {tab === 'runs' && (
        <div>
          {runs.length === 0 ? (
            <div className="text-center py-16 text-ink-faint text-sm">No execution runs yet.</div>
          ) : (
            <RunsTable runs={runs} />
          )}
        </div>
      )}

      {/* ── Context Tab ──────────────────────────────────────────────────────── */}
      {tab === 'context' && (
        <div className="max-w-3xl space-y-6">
          <div className="bg-surface-raised border border-line rounded-xl p-5">
            <h3 className="text-sm font-semibold text-ink mb-2">Project Instructions</h3>
            <p className="text-xs text-ink-faint mb-3">These instructions are prepended to every prompt in this project.</p>
            <textarea
              className="w-full bg-surface border border-line rounded-lg p-3 text-sm text-ink outline-none resize-y min-h-[100px] focus:border-accent/50"
              value={instrDraft}
              onChange={e => setInstrDraft(e.target.value)}
              placeholder="e.g. You are working on a Python 3.11 codebase. Always use type hints."
            />
            <div className="flex justify-end mt-2">
              <button
                onClick={handleSaveInstructions}
                className="flex items-center space-x-1.5 bg-accent/20 hover:bg-accent/30 text-accent px-4 py-2 rounded-lg text-sm font-medium border border-accent/30"
              >
                {savedInstr ? Icons.Check : Icons.Send}
                <span>{savedInstr ? 'Saved!' : 'Save Instructions'}</span>
              </button>
            </div>
          </div>

          <div className="bg-surface-raised border border-line rounded-xl p-5">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-semibold text-ink">Preview Context</h3>
              <button
                onClick={handleBuildContext}
                disabled={ctxLoading}
                className="flex items-center space-x-1.5 text-xs text-ink-muted hover:text-ink bg-surface-hover hover:bg-line-strong px-3 py-1.5 rounded-lg transition-colors"
              >
                {Icons.Refresh}
                <span>{ctxLoading ? 'Building…' : 'Build Context'}</span>
              </button>
            </div>
            <p className="text-xs text-ink-faint mb-4">
              Shows what context prefix would be prepended to the next prompt,
              using project instructions + recent conversation + {attachedFiles.length} attached file(s).
            </p>
            {ctxData && (
              <>
                <div className="grid grid-cols-1 gap-4 mb-4 text-xs sm:grid-cols-3">
                  <div className="bg-surface p-3 rounded-lg">
                    <p className="text-ink-faint mb-1">Tokens Used</p>
                    <p className="font-mono text-ink">{ctxData.total_tokens} / {ctxData.budget_tokens}</p>
                  </div>
                  <div className="bg-surface p-3 rounded-lg">
                    <p className="text-ink-faint mb-1">Pieces</p>
                    <p className="font-mono text-ink">{ctxData.pieces?.length ?? 0}</p>
                  </div>
                  <div className="bg-surface p-3 rounded-lg">
                    <p className="text-ink-faint mb-1">Truncated</p>
                    <p className={`font-mono ${ctxData.truncated ? 'text-warn' : 'text-ok'}`}>
                      {ctxData.truncated ? 'Yes' : 'No'}
                    </p>
                  </div>
                </div>
                {ctxData.truncation_warning && (
                  <div className="mb-3 text-xs text-warn bg-warn/10 border border-warn/20 rounded p-2">
                    {ctxData.truncation_warning}
                  </div>
                )}
                {ctxData.paid_provider_warning && (
                  <div className="mb-3 text-xs text-warn bg-warn/10 border border-warn/20 rounded p-2">
                    {ctxData.paid_provider_warning}
                  </div>
                )}
                <div className="bg-surface p-4 rounded-lg border border-line">
                  <p className="text-xs text-ink-faint mb-2 uppercase tracking-wider">Context Preview</p>
                  <pre className="text-xs text-ink font-mono whitespace-pre-wrap overflow-x-auto max-h-48 overflow-y-auto">
                    {ctxData.context_text || '(empty — no instructions, messages, or files selected)'}
                  </pre>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Settings Tab ─────────────────────────────────────────────────────── */}
      {tab === 'settings' && (
        <div className="max-w-xl space-y-6">
          <div className="bg-surface-raised border border-line rounded-xl p-5">
            <h3 className="text-sm font-semibold text-ink mb-4">Project Details</h3>
            <ProjectSettingsForm project={project} onUpdated={setProject} onError={setError} />
          </div>
          <div className="bg-surface-raised border border-danger/20 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-danger mb-2">Danger Zone</h3>
            <p className="text-xs text-ink-faint mb-3">Deleting a project removes all conversations, messages, files, and runs.</p>
            <button
              onClick={async () => {
                if (!confirm('Permanently delete this project?')) return;
                try {
                  await apiDelete(`/api/projects/${project.id}`);
                  onBack();
                } catch (err) {
                  setError(err.message);
                }
              }}
              className="bg-danger/10 hover:bg-danger/20 border border-danger/30 text-danger px-4 py-2 rounded-lg text-sm"
            >
              Delete Project
            </button>
          </div>
        </div>
      )}

      {pendingCost && (
        <CostConfirmationDialog
          preview={pendingCost}
          onPaid={() => { setPendingCost(null); handleExecute('paid'); }}
          onFree={() => { setPendingCost(null); handleExecute('free'); }}
          onCancel={() => setPendingCost(null)}
        />
      )}
    </div>
  );
}

// ── Runs table (reusable) ─────────────────────────────────────────────────────
function RunsTable({ runs }) {
  const [selected, setSelected] = useState(null);
  return (
    <>
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full min-w-[48rem] bg-surface-raised text-sm">
          <thead className="bg-surface text-ink-faint text-xs uppercase tracking-wider">
            <tr>
              <th className="text-left p-3 font-medium">Time</th>
              <th className="text-left p-3 font-medium">Model</th>
              <th className="text-left p-3 font-medium">Tier</th>
              <th className="text-right p-3 font-medium">Tokens</th>
              <th className="text-right p-3 font-medium">Cost</th>
              <th className="text-right p-3 font-medium">Latency</th>
              <th className="text-center p-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {runs.map(r => (
              <tr
                key={r.id}
                onClick={() => setSelected(selected?.id === r.id ? null : r)}
                className="hover:bg-surface-hover cursor-pointer"
              >
                <td className="p-3 text-ink-muted text-xs font-mono">{fmtDate(r.created_at).split(',')[0]}</td>
                <td className="p-3 text-accent text-xs font-mono truncate max-w-[180px]">{r.model ?? '—'}</td>
                <td className="p-3 text-ink text-xs font-mono">{r.tier ?? '—'}</td>
                <td className="p-3 text-right text-xs font-mono text-ink-muted">
                  {r.input_tokens != null ? `${r.input_tokens} / ${r.output_tokens}` : '—'}
                </td>
                <td className="p-3 text-right text-xs font-mono text-ok">
                  {r.actual_cost_usd != null ? `$${Number(r.actual_cost_usd).toFixed(5)}` : '—'}
                </td>
                <td className="p-3 text-right text-xs font-mono text-ink-muted">
                  {r.total_latency_ms != null ? `${Math.round(r.total_latency_ms)}ms` : '—'}
                </td>
                <td className="p-3 text-center">
                  <Badge color={r.status === 'success' ? 'emerald' : 'red'}>{r.status ?? '—'}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected && (
        <div className="mt-4 bg-surface border border-line rounded-xl p-5">
          <div className="flex justify-between items-center mb-3">
            <p className="text-xs text-ink-faint uppercase tracking-wider">Run Detail</p>
            <button onClick={() => setSelected(null)} className="text-ink-faint hover:text-ink text-sm">✕</button>
          </div>
          <pre className="text-xs text-ok font-mono whitespace-pre-wrap overflow-x-auto">
            {JSON.stringify(selected, null, 2)}
          </pre>
        </div>
      )}
    </>
  );
}

// ── Project settings form ─────────────────────────────────────────────────────
function ProjectSettingsForm({ project, onUpdated, onError }) {
  const [name, setName]   = useState(project.name);
  const [desc, setDesc]   = useState(project.description || '');
  const [saved, setSaved] = useState(false);

  async function handleSave(e) {
    e.preventDefault();
    const updated = await apiFetch(`/api/projects/${project.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name, description: desc }),
    }).catch(err => { onError(err.message); return null; });
    if (updated) {
      onUpdated(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }
  }

  return (
    <form onSubmit={handleSave} className="space-y-4">
      <div>
        <label className="block text-xs text-ink-muted mb-1">Name</label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink outline-none focus:border-accent/50"
        />
      </div>
      <div>
        <label className="block text-xs text-ink-muted mb-1">Description</label>
        <input
          value={desc}
          onChange={e => setDesc(e.target.value)}
          className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink outline-none focus:border-accent/50"
        />
      </div>
      <div className="flex justify-end">
        <button type="submit" className="bg-accent/20 hover:bg-accent/30 text-accent px-4 py-2 rounded-lg text-sm font-medium border border-accent/30">
          {saved ? 'Saved!' : 'Save Changes'}
        </button>
      </div>
    </form>
  );
}
