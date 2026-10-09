/**
 * Memory.jsx — the memory layer screen.
 *
 * Three concerns:
 *  1. List/manage the modular memory file (data/memory.json) — add, edit,
 *     delete, filter. The file is also hand-editable on disk; this screen is
 *     just the friendly view over it.
 *  2. Import conversation exports and keep each upload as a named collection
 *     that Chat can recall with an @mention.
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

async function apiFetch(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const res = await fetch(path, {
    headers: isFormData ? {} : { 'Content-Type': 'application/json' },
    cache: 'no-store',
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

const SOURCE_COLORS = {
  import:  '#38bdf8',
  manual:  '#94a3b8',
};
const sourceColor = source => SOURCE_COLORS[source] ?? '#94a3b8';

// ── Memory entry row ──────────────────────────────────────────────────────────

function displayMemoryTitle(entry) {
  const explicitTitle = entry.title?.trim() || entry.conversation_title?.trim();
  if (explicitTitle) return explicitTitle;
  const compactText = entry.text.replace(/\s+/g, ' ').trim();
  return compactText.length > 82 ? `${compactText.slice(0, 79)}…` : compactText || 'Untitled memory';
}

function MemoryRow({ entry, onSave, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState(entry.title || '');
  const [renameValue, setRenameValue] = useState(entry.title || '');
  const [text, setText] = useState(entry.text);
  const [tags, setTags] = useState(entry.tags.join(', '));
  const renameInputRef = useRef(null);

  useEffect(() => {
    if (editing || renaming) return;
    setTitle(entry.title || '');
    setRenameValue(entry.title || '');
    setText(entry.text);
    setTags(entry.tags.join(', '));
  }, [entry.id, entry.title, entry.text, entry.tags, editing, renaming]);

  function beginRename() {
    setRenameValue(entry.title || displayMemoryTitle(entry));
    setRenaming(true);
    setTimeout(() => {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }, 0);
  }

  function cancelRename() {
    setRenameValue(entry.title || '');
    setRenaming(false);
  }

  function submitRename(event) {
    event.preventDefault();
    const nextTitle = renameValue.trim();
    if (!nextTitle) return;
    onSave(entry.id, { title: nextTitle });
    setRenaming(false);
  }

  if (editing) {
    return (
      <div className="rounded-xl border border-accent/40 bg-surface-raised p-4 shadow-lg shadow-accent/5">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">Edit memory</p>
            <p className="mt-1 text-xs text-ink-faint">Give this piece of context a name you’ll recognize later.</p>
          </div>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="rounded-md px-2 py-1 text-xs text-ink-faint hover:bg-surface-hover hover:text-ink"
          >
            Close
          </button>
        </div>
        <label className="mb-2 block">
          <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-ink-faint">Name</span>
          <input
            value={title}
            onChange={e => setTitle(e.target.value)}
            placeholder="e.g. Preferred response style"
            className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
          />
        </label>
        <label className="mb-2 block">
          <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-ink-faint">Memory</span>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          rows={2}
            className="w-full resize-y rounded-lg border border-line bg-surface px-3 py-2 text-sm leading-relaxed text-ink outline-none focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
        />
        </label>
        <label className="block">
          <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-ink-faint">Tags</span>
          <input
            value={tags}
            onChange={e => setTags(e.target.value)}
            placeholder="preferences, workflow, projects"
            className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-xs font-mono text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
          />
        </label>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            onClick={() => {
              onSave(entry.id, {
                title: title.trim(),
                text,
                tags: tags.split(',').map(t => t.trim()).filter(Boolean),
              });
              setEditing(false);
            }}
            className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent hover:bg-accent-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
          >
            Save changes
          </button>
          <button
            onClick={() => { setText(entry.text); setTags(entry.tags.join(', ')); setEditing(false); }}
            className="rounded-lg px-3 py-1.5 text-xs text-ink-muted hover:bg-surface-hover hover:text-ink"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <article className="group rounded-xl border border-line bg-surface-raised/80 p-4 transition-colors hover:border-line-strong hover:bg-surface-raised">
      <div className="flex items-start gap-3">
        <span
          className="mt-2 h-2 w-2 flex-shrink-0 rounded-full ring-4 ring-surface-raised"
          style={{ backgroundColor: sourceColor(entry.source) }}
          title={entry.source}
        />
        <div className="min-w-0 flex-1">
          {renaming ? (
            <form onSubmit={submitRename} className="flex max-w-2xl flex-wrap items-center gap-2">
              <input
                ref={renameInputRef}
                value={renameValue}
                onChange={e => setRenameValue(e.target.value)}
                onKeyDown={e => { if (e.key === 'Escape') cancelRename(); }}
                aria-label="Memory name"
                className="min-w-[12rem] flex-1 rounded-lg border border-accent/60 bg-surface px-3 py-1.5 text-sm font-medium text-ink outline-none focus:ring-2 focus:ring-accent/20"
              />
              <button type="submit" className="rounded-lg bg-accent px-2.5 py-1.5 text-xs font-semibold text-on-accent hover:bg-accent-strong">Save</button>
              <button type="button" onClick={cancelRename} className="rounded-lg px-2.5 py-1.5 text-xs text-ink-muted hover:bg-surface-hover hover:text-ink">Cancel</button>
            </form>
          ) : (
            <div className="flex min-w-0 items-start gap-2">
              <button
                type="button"
                onClick={beginRename}
                title="Rename memory"
                className="min-w-0 text-left text-sm font-semibold leading-6 text-ink hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              >
                <span className="break-words">{displayMemoryTitle(entry)}</span>
              </button>
              <button
                type="button"
                onClick={beginRename}
                aria-label={`Rename ${displayMemoryTitle(entry)}`}
                title="Rename memory"
                className="mt-1 flex-shrink-0 rounded-md p-1 text-ink-faint opacity-100 transition-opacity hover:bg-surface-hover hover:text-accent sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100"
              >
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                  <path d="M11.5 2.5a1.414 1.414 0 012 2L5 13H2v-3L11.5 2.5z" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          )}
          <p className="mt-1 max-w-3xl break-words text-sm leading-relaxed text-ink-muted">{entry.text}</p>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <span className="rounded-full border border-line px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-ink-faint">{entry.source}</span>
            {entry.tags.map(tag => (
              <span key={tag} className="rounded-full bg-surface px-2 py-0.5 text-[10px] font-mono text-ink-muted">
                #{tag}
              </span>
            ))}
            {entry.conversation_title && entry.title && (
              <span className="max-w-full truncate text-[10px] italic text-ink-faint">
                from “{entry.conversation_title}”
              </span>
            )}
          </div>
        </div>
        <div className="flex flex-shrink-0 items-center gap-1 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">
          <button
            onClick={() => setEditing(true)}
            aria-label="Edit memory"
            title="Edit memory"
            className="rounded-lg p-2 text-ink-faint hover:bg-surface-hover hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
          >
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M11.5 2.5a1.414 1.414 0 012 2L5 13H2v-3L11.5 2.5z" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button
            onClick={() => onDelete(entry.id)}
            aria-label="Delete memory"
            title="Delete memory"
            className="rounded-lg p-2 text-ink-faint hover:bg-danger/10 hover:text-danger focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50"
          >
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M3 4.5h10M6 4.5V3h4v1.5m-5.5 0 .5 9h6l.5-9M7 7v4m2-4v4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </div>
    </article>
  );
}

function MemoryUploadRow({ upload, onRename }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(upload.name);
  const sources = (upload.sources || []).filter(source => source !== 'import');

  useEffect(() => {
    if (!editing) setValue(upload.name);
  }, [upload.name, editing]);

  function submit(event) {
    event.preventDefault();
    const next = value.trim().toLowerCase();
    if (!next) return;
    onRename(upload.id, next);
    setEditing(false);
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-surface-raised/70 px-3 py-2.5">
      <div className="min-w-0">
        {editing ? (
          <form onSubmit={submit} className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs text-accent">@</span>
            <input
              autoFocus
              value={value}
              onChange={event => setValue(event.target.value)}
              onKeyDown={event => { if (event.key === 'Escape') setEditing(false); }}
              aria-label={`Rename ${upload.name}`}
              className="w-40 rounded-lg border border-accent/60 bg-surface px-2 py-1 text-xs font-mono text-ink outline-none"
            />
            <button type="submit" className="rounded-lg bg-accent px-2.5 py-1 text-xs font-semibold text-on-accent">Save</button>
            <button type="button" onClick={() => setEditing(false)} className="rounded-lg px-2 py-1 text-xs text-ink-muted hover:bg-surface-hover">Cancel</button>
          </form>
        ) : (
          <button type="button" onClick={() => setEditing(true)} className="group flex items-center gap-2 text-left">
            <span className="font-mono text-sm text-accent">@{upload.name}</span>
            <span className="text-[10px] text-ink-faint group-hover:text-ink-muted">Rename</span>
          </button>
        )}
        <p className="mt-1 text-[11px] text-ink-faint">
          {upload.count.toLocaleString()} {upload.count === 1 ? 'memory' : 'memories'}
          {sources.length ? ` · ${sources.join(', ')}` : ''}
        </p>
      </div>
      {!editing && (
        <span className="rounded-full border border-line px-2 py-0.5 text-[10px] text-ink-faint">Use @{upload.name} in Chat</span>
      )}
    </div>
  );
}

// ── Main screen ───────────────────────────────────────────────────────────────

const SUPPORTED_FORMATS = [
  { ext: '.json', label: 'JSON', desc: 'Conversation exports or generic memory JSON' },
  { ext: '.csv',  label: 'CSV',  desc: 'Rows with a "text" column (tags, source optional)' },
  { ext: '.txt',  label: 'TXT',  desc: 'Plain text — one memory per line' },
  { ext: '.md',   label: 'MD',   desc: 'Markdown — each paragraph or list item becomes a memory' },
  { ext: '.zip',  label: 'ZIP',  desc: 'Conversation export or any folder packed as an archive' },
];
// Folder selections cannot be constrained consistently across browsers, so
// the folder picker stays open-ended. Direct file selections get a useful
// export-oriented filter while still preserving JSON and Markdown imports.
const ACCEPTED_IMPORT_TYPES = '.zip,.json,.md,.markdown,.txt,.csv,.tsv,.jsonl';

export default function MemoryScreen() {
  const [memories, setMemories] = useState([]);
  const [filePath, setFilePath] = useState('');
  const [uploads, setUploads] = useState([]);
  const [importName, setImportName] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;
  const [memoryName, setMemoryName] = useState('memory');
  const [isRenaming, setIsRenaming] = useState(false);
  const [renamingValue, setRenamingValue] = useState('');
  const [renameError, setRenameError] = useState(null);
  const renameInputRef = useRef(null);
  const [newTitle, setNewTitle] = useState('');
  const [newText, setNewText] = useState('');
  const [newTags, setNewTags] = useState('');
  const [importResult, setImportResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const newTextRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (sourceFilter) params.set('source', sourceFilter);
      if (search) params.set('q', search);
      const data = await apiFetch(`/api/memory?${params}`);
      setMemories(data.memories);
      setFilePath(data.file_path);
      setUploads(data.uploads ?? []);
      if (data.name) setMemoryName(data.name);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [sourceFilter, search]);

  useEffect(() => { load(); }, [load]);

  // Debounce search input — only fire API call 300ms after typing stops
  useEffect(() => {
    const t = setTimeout(() => setSearch(searchInput), 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  // Reset to first page whenever filters change
  useEffect(() => { setPage(0); }, [search, sourceFilter]);

  async function addMemory() {
    if (!newText.trim()) return;
    try {
      await apiFetch('/api/memory', {
        method: 'POST',
        body: JSON.stringify({
          title: newTitle,
          text: newText,
          tags: newTags.split(',').map(t => t.trim()).filter(Boolean),
        }),
      });
      setNewTitle('');
      setNewText('');
      setNewTags('');
      load();
    } catch (e) {
      setError(e.message);
    }
  }

  async function saveMemory(id, patch) {
    try {
      await apiFetch(`/api/memory/${id}`, { method: 'PUT', body: JSON.stringify(patch) });
      load();
    } catch (e) { setError(e.message); }
  }

  async function deleteMemory(id) {
    try {
      await apiFetch(`/api/memory/${id}`, { method: 'DELETE' });
      load();
    } catch (e) { setError(e.message); }
  }

  async function renameUpload(id, name) {
    try {
      await apiFetch(`/api/memory/uploads/${id}`, {
        method: 'PATCH',
        body: JSON.stringify({ name }),
      });
      load();
    } catch (e) {
      setError(e.message);
    }
  }

  async function saveMemoryName() {
    const name = renamingValue.trim().toLowerCase();
    if (!name) return;
    setRenameError(null);
    try {
      const data = await apiFetch('/api/memory/name', {
        method: 'PATCH',
        body: JSON.stringify({ name }),
      });
      setMemoryName(data.name);
      setIsRenaming(false);
    } catch (e) {
      setRenameError(e.message);
    }
  }

  function startRenaming() {
    setRenamingValue(memoryName);
    setRenameError(null);
    setIsRenaming(true);
    // Focus after render
    setTimeout(() => renameInputRef.current?.focus(), 0);
  }

  async function importFiles(files) {
    setImportResult(null);
    setError(null);
    try {
      const form = new FormData();
      for (const file of files) form.append('files', file, file.webkitRelativePath || file.name);
      if (importName.trim()) form.append('name', importName.trim());
      const result = await apiFetch('/api/memory/import/files', { method: 'POST', body: form });
      setImportResult(result);
      load();
    } catch (e) {
      setError(e.message.startsWith('API') ? e.message : `Could not import files: ${e.message}`);
    }
  }

  function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }
  function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    if (dropZoneRef.current && !dropZoneRef.current.contains(e.relatedTarget)) {
      setDragging(false);
    }
  }
  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length > 0) importFiles(files);
  }

  function focusComposer() {
    setTimeout(() => newTextRef.current?.focus(), 0);
  }

  const sources = useMemo(() => [...new Set(memories.map(m => m.source))], [memories]);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-5xl space-y-5 px-4 py-5 sm:px-6 lg:px-8">
        {/* Header: the page is a workspace, not a file browser. */}
        <header className="relative overflow-hidden rounded-2xl border border-line bg-surface-raised/70 p-5 sm:p-6">
          <div className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-accent/10 blur-3xl" />
          <div className="relative flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div className="min-w-0">
              <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-accent">Persistent context</p>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Memory</h1>
                <span className="rounded-full border border-accent/25 bg-accent/10 px-2.5 py-1 text-[11px] font-medium text-accent">
                  {memories.length.toLocaleString()} visible
                </span>
              </div>
              <p className="mt-2 max-w-xl text-sm leading-relaxed text-ink-muted">
                Keep the context worth carrying forward. Name each memory so it stays useful long after the original conversation is gone.
              </p>
              <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-ink-faint">Store</span>
                {isRenaming ? (
                  <form onSubmit={e => { e.preventDefault(); saveMemoryName(); }} className="flex flex-wrap items-center gap-1.5">
                    <span className="font-mono text-accent">@</span>
                    <input
                      ref={renameInputRef}
                      value={renamingValue}
                      onChange={e => setRenamingValue(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Escape') { setIsRenaming(false); setRenameError(null); } }}
                      placeholder="memory-name"
                      aria-label="Memory store name"
                      className="w-32 rounded-lg border border-accent/60 bg-surface px-2 py-1 font-mono text-xs text-ink outline-none focus:ring-2 focus:ring-accent/20"
                    />
                    <button type="submit" className="rounded-lg bg-accent px-2.5 py-1 text-xs font-semibold text-on-accent hover:bg-accent-strong">Save</button>
                    <button type="button" onClick={() => { setIsRenaming(false); setRenameError(null); }} className="rounded-lg px-2 py-1 text-xs text-ink-muted hover:bg-surface-hover hover:text-ink">Cancel</button>
                    {renameError && <span className="basis-full text-xs text-danger sm:basis-auto">{renameError}</span>}
                  </form>
                ) : (
                  <button
                    type="button"
                    onClick={startRenaming}
                    title="Rename the @mention alias for this memory store"
                    className="group inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1 font-mono text-accent/90 hover:border-accent/50 hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
                  >
                    @{memoryName}
                    <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                      <path d="M11.5 2.5a1.414 1.414 0 012 2L5 13H2v-3L11.5 2.5z" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span className="font-sans text-[10px] text-ink-faint group-hover:text-ink-muted">Rename</span>
                  </button>
                )}
                <span className="text-ink-faint">·</span>
                <code className="max-w-full truncate text-[11px] text-ink-faint" title={filePath || 'data/memory.json'}>{filePath || 'data/memory.json'}</code>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={focusComposer}
                className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-on-accent shadow-lg shadow-accent/10 hover:bg-accent-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
              >
                <span className="text-base leading-none">+</span> Add memory
              </button>
              <button
                type="button"
                onClick={() => folderInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-accent/35 bg-accent/10 px-3 py-2 text-xs font-semibold text-accent hover:border-accent/60 hover:bg-accent/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                  <path d="M3.5 7.5h6l1.8 2H20a1 1 0 0 1 1 1v7.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M12 12v5m0 0 2-2m-2 2-2-2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Import export
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded-lg border border-line bg-surface px-3 py-2 text-xs font-medium text-ink-muted hover:border-accent/40 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              >
                ZIP or files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_IMPORT_TYPES}
                multiple
                className="hidden"
                onChange={e => {
                  importFiles(Array.from(e.target.files || []));
                  e.target.value = '';
                }}
              />
              <input
                ref={folderInputRef}
                type="file"
                webkitdirectory="true"
                directory="true"
                multiple
                className="hidden"
                onChange={e => {
                  importFiles(Array.from(e.target.files || []));
                  e.target.value = '';
                }}
              />
            </div>
          </div>
        </header>

      {error && (
        <div role="alert" className="rounded-xl border border-danger/30 bg-danger/10 p-3 text-xs text-danger break-words">
          {error}
        </div>
      )}
      {importResult && (
        <div role="status" className="rounded-xl border border-ok/30 bg-ok/10 p-3 text-xs text-ok">
          Imported {importResult.memories_added} memories
          {importResult.upload?.name ? ` into @${importResult.upload.name}` : ''}
          ({importResult.conversations_parsed} conversations parsed
          {importResult.duplicates_skipped > 0 ? `, ${importResult.duplicates_skipped} duplicates skipped` : ''}
          {importResult.files_skipped > 0 ? `, ${importResult.files_skipped} unsupported files skipped` : ''}).
        </div>
      )}
      {uploads.length > 0 && (
        <section className="rounded-2xl border border-line bg-surface-raised/45 p-4" aria-label="Memory uploads">
          <div className="mb-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">Memory uploads</p>
            <p className="mt-1 text-xs text-ink-muted">Each upload has a distinct alias. Use <span className="font-mono">@alias</span> in Chat to recall only that collection.</p>
          </div>
          <div className="space-y-2">
            {uploads.map(upload => (
              <MemoryUploadRow key={upload.id} upload={upload} onRename={renameUpload} />
            ))}
          </div>
        </section>
      )}

      <>
          {/* Bring in an export, or add one memory by hand. */}
          <div
            ref={dropZoneRef}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`rounded-2xl border bg-surface-raised p-4 transition-colors sm:p-5 ${
              dragging
                ? 'border-accent border-dashed bg-accent/5'
                : 'border-line'
            }`}
          >
            {dragging ? (
              <div className="flex flex-col items-center justify-center py-8 gap-2">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-accent">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
                  <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
                  <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" />
                </svg>
                <p className="text-sm font-medium text-accent">Drop the folder or ZIP to import memories</p>
                <p className="text-[11px] text-ink-muted">
                  JSON and Markdown files inside are included automatically
                </p>
              </div>
            ) : (
              <>
                <div className="mb-4 rounded-xl border border-dashed border-accent/35 bg-accent/5 p-3.5 sm:p-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="min-w-0">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">Import a conversation export</p>
                      <p className="mt-1 text-sm font-medium text-ink">Bring the whole folder or its ZIP archive</p>
                      <p className="mt-1 max-w-xl text-xs leading-relaxed text-ink-muted">
                        Drop it here, choose a folder, or select a ZIP. Direct JSON and Markdown files are supported too.
                      </p>
                      <label className="mt-3 block max-w-xs">
                        <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-ink-faint">Import name (optional)</span>
                        <input
                          value={importName}
                          onChange={e => setImportName(e.target.value)}
                          placeholder="e.g. personal, work, research"
                          aria-label="Import name"
                          className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-xs font-mono text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
                        />
                      </label>
                    </div>
                    <div className="flex shrink-0 flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => folderInputRef.current?.click()}
                        className="inline-flex items-center justify-center rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-on-accent hover:bg-accent-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
                      >
                        Choose folder
                      </button>
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="inline-flex items-center justify-center rounded-lg border border-line bg-surface px-3 py-2 text-xs font-medium text-ink-muted hover:border-accent/40 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
                      >
                        Choose ZIP or files
                      </button>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {SUPPORTED_FORMATS.map(f => (
                      <span
                        key={f.ext}
                        title={f.desc}
                        className="cursor-default rounded border border-line bg-surface px-1.5 py-0.5 text-[10px] font-mono text-ink-faint"
                      >
                        {f.label}
                      </span>
                    ))}
                    <span className="rounded border border-line bg-surface px-1.5 py-0.5 text-[10px] font-mono text-ink-faint">FOLDER</span>
                  </div>
                </div>
                <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">Add context</p>
                    <p className="mt-1 text-xs text-ink-faint">Name it now, or rename it later from the list.</p>
                  </div>
                  <span className="font-mono text-[10px] text-ink-faint">⌘↵ / Ctrl↵ to save</span>
                </div>
                <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_15rem]">
                  <textarea
                    ref={newTextRef}
                    value={newText}
                    onChange={e => setNewText(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) addMemory(); }}
                    placeholder={'What should Theriad remember? e.g. “I prefer concise answers with code examples.”'}
                    rows={3}
                    aria-label="New memory"
                    className="w-full resize-y rounded-xl border border-line bg-surface px-3 py-2.5 text-sm leading-relaxed text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
                  />
                  <div className="flex flex-col gap-2">
                    <input
                      value={newTitle}
                      onChange={e => setNewTitle(e.target.value)}
                      placeholder="Name (optional)"
                      aria-label="Memory name"
                      className="w-full rounded-xl border border-line bg-surface px-3 py-2 text-xs text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
                    />
                    <input
                      value={newTags}
                      onChange={e => setNewTags(e.target.value)}
                      placeholder="Tags, comma separated"
                      aria-label="Memory tags"
                      className="w-full rounded-xl border border-line bg-surface px-3 py-2 text-xs font-mono text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
                    />
                    <button
                      type="button"
                      onClick={addMemory}
                      disabled={!newText.trim()}
                      className="mt-auto inline-flex items-center justify-center rounded-xl bg-accent px-4 py-2 text-xs font-semibold text-on-accent hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
                    >
                      Save memory
                    </button>
                  </div>
                </div>
                {/* Drop hint */}
                <div className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[11px] text-ink-faint">
                  <div className="flex items-center gap-1.5">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="opacity-50">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
                      <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
                      <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" />
                    </svg>
                    Drop a folder or ZIP here to import, or use the buttons above
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Search and filters */}
          <section className="rounded-2xl border border-line bg-surface-raised/45 p-3 sm:p-4" aria-label="Memory filters">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <label className="relative min-w-0 flex-1">
                <span className="sr-only">Search memories</span>
                <svg className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint" width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <circle cx="7" cy="7" r="4.5" />
                  <path d="m10.5 10.5 3 3" strokeLinecap="round" />
                </svg>
                <input
                  value={searchInput}
                  onChange={e => setSearchInput(e.target.value)}
                  placeholder="Search names, memory text, or tags…"
                  className="w-full rounded-xl border border-line bg-surface py-2.5 pl-9 pr-9 text-sm text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/20"
                />
                {searchInput && (
                  <button
                    type="button"
                    onClick={() => setSearchInput('')}
                    aria-label="Clear search"
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-ink-faint hover:bg-surface-hover hover:text-ink"
                  >
                    ×
                  </button>
                )}
              </label>
              <p className="shrink-0 text-xs text-ink-faint">
                {memories.length.toLocaleString()} {memories.length === 1 ? 'result' : 'results'}
              </p>
            </div>
            <div className="mt-3 flex min-w-0 gap-1.5 overflow-x-auto pb-0.5">
              <button
                type="button"
                onClick={() => setSourceFilter('')}
                className={`shrink-0 rounded-lg px-2.5 py-1.5 text-xs font-medium ${!sourceFilter ? 'bg-accent/15 text-accent' : 'text-ink-muted hover:bg-surface-hover hover:text-ink'}`}
              >
                All sources
              </button>
              {sources.map(s => (
                <button
                  type="button"
                  key={s}
                  onClick={() => setSourceFilter(s === sourceFilter ? '' : s)}
                  className={`flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium ${
                    sourceFilter === s ? 'bg-accent/15 text-accent' : 'text-ink-muted hover:bg-surface-hover hover:text-ink'
                  }`}
                >
                  <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: sourceColor(s) }} />
                  {s}
                </button>
              ))}
            </div>
          </section>

          {/* List */}
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-faint">Your library</p>
              <p className="mt-1 text-xs text-ink-muted">Click a memory name to rename it instantly.</p>
            </div>
            {sourceFilter && <span className="rounded-full border border-line bg-surface-raised px-2.5 py-1 text-[10px] font-mono text-ink-faint">source:{sourceFilter}</span>}
          </div>
          <div className="space-y-2">
            {memories.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(m => (
              <MemoryRow key={m.id} entry={m} onSave={saveMemory} onDelete={deleteMemory} />
            ))}
            {memories.length === 0 && (
              <div className="rounded-2xl border border-dashed border-line p-10 text-center">
                <p className="text-sm font-medium text-ink">{search || sourceFilter ? 'Nothing matches these filters' : 'Your memory library is empty'}</p>
                <p className="mx-auto mt-2 max-w-md text-xs leading-relaxed text-ink-muted">
                  {search || sourceFilter
                    ? 'Try a different search or clear the active source filter.'
                    : 'Start with one durable preference, workflow detail, or project fact. You can import larger archives whenever you need.'}
                </p>
                {(search || sourceFilter) ? (
                  <button type="button" onClick={() => { setSearchInput(''); setSourceFilter(''); }} className="mt-4 rounded-lg border border-line bg-surface-raised px-3 py-1.5 text-xs font-medium text-ink-muted hover:border-accent/40 hover:text-ink">Clear filters</button>
                ) : (
                  <button type="button" onClick={focusComposer} className="mt-4 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent hover:bg-accent-strong">Add your first memory</button>
                )}
              </div>
            )}
          </div>

          {/* Pagination */}
          {memories.length > PAGE_SIZE && (
            <div className="flex items-center justify-between text-xs text-ink-muted py-1">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1.5 bg-surface-raised border border-line rounded-lg hover:border-accent/40 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                ← Previous
              </button>
              <span>
                {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, memories.length)} of {memories.length.toLocaleString()}
              </span>
              <button
                onClick={() => setPage(p => Math.min(Math.ceil(memories.length / PAGE_SIZE) - 1, p + 1))}
                disabled={(page + 1) * PAGE_SIZE >= memories.length}
                className="px-3 py-1.5 bg-surface-raised border border-line rounded-lg hover:border-accent/40 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Next →
              </button>
            </div>
          )}

      </>
    </div>
    </div>
  );
}
