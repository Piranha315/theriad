import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toFriendlyError } from './uiErrors.js';
import { useDelayedLoading } from './components/useDelayedLoading.js';

const ACCEPTED_FILES = '.md,.txt,.zip,text/markdown,text/plain,application/zip';

async function apiFetch(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const { headers: extraHeaders, ...rest } = options;
  const response = await fetch(path, {
    headers: isFormData ? { ...extraHeaders } : { 'Content-Type': 'application/json', ...extraHeaders },
    ...rest,
  });
  const raw = response.status === 204 ? '' : await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = { error: raw };
    }
  }
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.error ?? response.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return payload;
}

function formatDate(value) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function SkillSkeletons() {
  return (
    <div className="space-y-3" aria-label="Loading skills">
      {[0, 1, 2].map(item => (
        <div key={item} className="rounded-xl border border-line bg-surface-raised p-5">
          <div className="theriad-skeleton-block h-4 w-40 rounded" />
          <div className="theriad-skeleton-block mt-3 h-3 w-3/4 rounded" />
          <div className="theriad-skeleton-block mt-2 h-3 w-1/2 rounded" />
        </div>
      ))}
    </div>
  );
}

function Toggle({ checked, disabled, onChange, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onChange}
      className={`relative h-6 w-11 flex-shrink-0 rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:cursor-not-allowed disabled:opacity-40 ${
        checked ? 'border-accent bg-accent' : 'border-line-strong bg-surface-sunken'
      }`}
    >
      <span
        aria-hidden="true"
        className={`absolute left-1 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`}
      />
    </button>
  );
}

export default function SkillsScreen() {
  const [skills, setSkills] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [importing, setImporting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [busyIds, setBusyIds] = useState({});
  const [confirmingDeleteId, setConfirmingDeleteId] = useState(null);
  const [query, setQuery] = useState('');
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const showSkeleton = useDelayedLoading(loading);

  const loadSkills = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    setError(null);
    try {
      const data = await apiFetch('/api/skills');
      setSkills(Array.isArray(data) ? data : data?.skills ?? []);
    } catch (loadError) {
      setError(toFriendlyError(loadError));
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSkills();
  }, [loadSkills]);

  const visibleSkills = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return skills;
    return skills.filter(skill => (
      `${skill.name ?? ''} ${skill.description ?? ''} ${skill.source_name ?? ''}`
        .toLowerCase()
        .includes(normalizedQuery)
    ));
  }, [query, skills]);

  async function importFiles(fileList) {
    const files = Array.from(fileList ?? []);
    if (files.length === 0 || importing) return;
    const accepted = files.filter(file => {
      const name = file.name.toLowerCase();
      return name.endsWith('.md') || name.endsWith('.txt') || name.endsWith('.zip');
    });
    if (accepted.length === 0) {
      setError('Choose a Markdown or text skill file, a skill folder, or a ZIP package containing SKILL.md.');
      return;
    }

    setImporting(true);
    setError(null);
    setNotice(null);
    try {
      const form = new FormData();
      accepted.forEach(file => form.append('files', file, file.webkitRelativePath || file.name));
      form.append('enabled', 'false');
      const result = await apiFetch('/api/skills/import/files', {
        method: 'POST',
        body: form,
      });
      const created = result?.skills_created ?? result?.skills?.length ?? 0;
      const duplicateCount = result?.duplicates_skipped ?? 0;
      const skippedCount = result?.files_skipped ?? files.length - accepted.length;
      const details = [];
      if (duplicateCount) details.push(`${duplicateCount} duplicate${duplicateCount === 1 ? '' : 's'} skipped`);
      if (skippedCount) details.push(`${skippedCount} unsupported file${skippedCount === 1 ? '' : 's'} skipped`);
      setNotice({
        text: `${created} skill${created === 1 ? '' : 's'} imported for review${details.length ? `. ${details.join(', ')}.` : '.'}`,
        warnings: Array.isArray(result?.warnings) ? result.warnings : [],
      });
      await loadSkills({ quiet: true });
    } catch (importError) {
      setError(toFriendlyError(importError));
    } finally {
      setImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (folderInputRef.current) folderInputRef.current.value = '';
    }
  }

  async function toggleSkill(skill) {
    if (busyIds[skill.id]) return;
    setBusyIds(current => ({ ...current, [skill.id]: true }));
    setError(null);
    try {
      const result = await apiFetch(`/api/skills/${encodeURIComponent(skill.id)}`, {
        method: 'PATCH',
        body: JSON.stringify({ enabled: !skill.enabled }),
      });
      const updated = result?.skill ?? { ...skill, enabled: !skill.enabled };
      setSkills(current => current.map(item => item.id === skill.id ? updated : item));
    } catch (toggleError) {
      setError(toFriendlyError(toggleError));
    } finally {
      setBusyIds(current => {
        const next = { ...current };
        delete next[skill.id];
        return next;
      });
    }
  }

  async function deleteSkill(skillId) {
    if (busyIds[skillId]) return;
    setBusyIds(current => ({ ...current, [skillId]: true }));
    setError(null);
    try {
      await apiFetch(`/api/skills/${encodeURIComponent(skillId)}`, { method: 'DELETE' });
      setSkills(current => current.filter(skill => skill.id !== skillId));
      setConfirmingDeleteId(null);
    } catch (deleteError) {
      setError(toFriendlyError(deleteError));
    } finally {
      setBusyIds(current => {
        const next = { ...current };
        delete next[skillId];
        return next;
      });
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);
    importFiles(event.dataTransfer?.files);
  }

  return (
    <div className="mx-auto max-w-5xl space-y-5 pb-12">
      <section className="overflow-hidden rounded-xl border border-line bg-surface-raised">
        <div className="grid gap-5 p-5 sm:p-6 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold text-ink">Reusable instructions for Theriad</p>
            <p className="mt-1 text-sm leading-relaxed text-ink-muted">
              Import a Markdown skill or a ZIP package. New skills stay disabled until you review and enable them.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 md:justify-end">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={importing}
              className="rounded-lg border border-accent/30 bg-accent/10 px-4 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {importing ? 'Importing...' : 'Import files'}
            </button>
            <button
              type="button"
              onClick={() => folderInputRef.current?.click()}
              disabled={importing}
              className="rounded-lg border border-line bg-surface px-4 py-2 text-sm font-medium text-ink-muted transition-colors hover:border-line-strong hover:bg-surface-hover hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Import folder
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_FILES}
            multiple
            className="hidden"
            onChange={event => importFiles(event.target.files)}
          />
          <input
            ref={folderInputRef}
            type="file"
            webkitdirectory="true"
            directory="true"
            multiple
            className="hidden"
            onChange={event => importFiles(event.target.files)}
          />
        </div>
        <div
          onDragOver={event => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={event => {
            event.preventDefault();
            if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false);
          }}
          onDrop={handleDrop}
          className={`border-t px-5 py-3 text-center text-xs transition-colors sm:px-6 ${
            dragging
              ? 'border-accent bg-accent/10 text-accent'
              : 'border-line bg-surface text-ink-faint'
          }`}
        >
          {dragging ? 'Drop skill files to import them' : 'You can also drop .md, .txt, or .zip skill files here'}
        </div>
      </section>

      <div aria-live="polite" className="space-y-3">
        {error && (
          <div className="flex flex-col justify-between gap-3 rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 sm:flex-row sm:items-center" role="alert">
            <p className="break-words text-sm text-danger">{error}</p>
            <button
              type="button"
              onClick={() => loadSkills()}
              className="self-start rounded-lg border border-danger/30 px-3 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10"
            >
              Reload
            </button>
          </div>
        )}
        {notice && (
          <div className="rounded-xl border border-ok/30 bg-ok/10 px-4 py-3 text-sm text-ok" role="status">
            <p>{notice.text}</p>
            {notice.warnings.length > 0 && (
              <details className="mt-2 text-xs text-ink-muted">
                <summary className="cursor-pointer">Import warnings</summary>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {notice.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
                </ul>
              </details>
            )}
          </div>
        )}
      </div>

      {showSkeleton && loading ? <SkillSkeletons /> : loading ? null : skills.length === 0 ? (
        <section className="rounded-xl border border-dashed border-line bg-surface-raised px-6 py-14 text-center">
          <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl border border-line bg-surface text-accent">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden="true">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
              <path d="M9 7h6M9 11h4" />
            </svg>
          </div>
          <h2 className="mt-4 text-base font-semibold text-ink">No skills imported</h2>
          <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-ink-muted">
            Import a skill package to give chat and agent runs reusable guidance for specific kinds of work.
          </p>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="mt-5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-on-accent transition-colors hover:bg-accent-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
          >
            Choose a skill file
          </button>
        </section>
      ) : (
        <section className="space-y-3">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
            <p className="text-xs text-ink-faint">
              {skills.filter(skill => skill.enabled).length} enabled of {skills.length} imported
            </p>
            {skills.length > 4 && (
              <label className="relative min-w-0 sm:w-64">
                <span className="sr-only">Search skills</span>
                <input
                  type="search"
                  value={query}
                  onChange={event => setQuery(event.target.value)}
                  placeholder="Search skills"
                  className="w-full rounded-lg border border-line bg-surface-raised px-3 py-2 text-sm text-ink outline-none placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/30"
                />
              </label>
            )}
          </div>

          {visibleSkills.length === 0 ? (
            <div className="rounded-xl border border-line bg-surface-raised p-8 text-center text-sm text-ink-muted">
              No skills match “{query}”.
            </div>
          ) : visibleSkills.map(skill => {
            const rowBusy = Boolean(busyIds[skill.id]);
            const deleting = confirmingDeleteId === skill.id;
            const updatedAt = formatDate(skill.updated_at ?? skill.created_at);
            return (
              <article key={skill.id} className="min-w-0 overflow-hidden rounded-xl border border-line bg-surface-raised p-5 transition-colors hover:border-line-strong">
                <div className="flex min-w-0 flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="break-words text-sm font-semibold text-ink">{skill.name}</h2>
                      <span className={`rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${
                        skill.enabled
                          ? 'border-ok/30 bg-ok/10 text-ok'
                          : 'border-line bg-surface text-ink-faint'
                      }`}>
                        {skill.enabled ? 'Enabled' : 'Review needed'}
                      </span>
                    </div>
                    <p className="mt-1 max-w-2xl break-words text-sm leading-relaxed text-ink-muted">
                      {skill.description || 'No description was included with this skill.'}
                    </p>
                    <p className="mt-2 flex min-w-0 flex-wrap gap-x-3 gap-y-1 text-[11px] text-ink-faint">
                      {skill.source_name && <span className="min-w-0 max-w-full break-all font-mono">{skill.source_name}</span>}
                      {skill.source_type && <span>{skill.source_type}</span>}
                      {Number.isFinite(skill.priority) && <span>Priority {skill.priority}</span>}
                      {updatedAt && <span>Updated {updatedAt}</span>}
                    </p>
                  </div>
                  <div className="flex flex-shrink-0 flex-wrap items-center gap-2 sm:justify-end">
                    <span className="text-xs text-ink-muted">Apply</span>
                    <Toggle
                      checked={Boolean(skill.enabled)}
                      disabled={rowBusy}
                      onChange={() => toggleSkill(skill)}
                      label={`${skill.enabled ? 'Disable' : 'Enable'} ${skill.name}`}
                    />
                  </div>
                </div>

                <div className="mt-4 flex flex-col justify-between gap-3 border-t border-line pt-3 sm:flex-row sm:items-center">
                  <details className="min-w-0 flex-1">
                    <summary className="cursor-pointer text-xs font-medium text-ink-muted hover:text-ink">Review instructions</summary>
                    <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-surface-sunken p-3 text-xs leading-relaxed text-ink-muted">{skill.instructions}</pre>
                  </details>
                  {deleting ? (
                    <div className="flex items-center gap-2 self-start">
                      <span className="text-xs text-danger">Remove this skill?</span>
                      <button
                        type="button"
                        onClick={() => deleteSkill(skill.id)}
                        disabled={rowBusy}
                        className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-1.5 text-xs font-medium text-danger hover:bg-danger/20 disabled:opacity-40"
                      >
                        {rowBusy ? 'Removing...' : 'Remove'}
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmingDeleteId(null)}
                        disabled={rowBusy}
                        className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink-muted hover:bg-surface-hover disabled:opacity-40"
                      >
                        Keep
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setConfirmingDeleteId(skill.id)}
                      className="self-start text-xs text-ink-faint transition-colors hover:text-danger"
                    >
                      Remove
                    </button>
                  )}
                </div>
              </article>
            );
          })}
        </section>
      )}

      <p className="rounded-xl border border-line bg-surface px-4 py-3 text-xs leading-relaxed text-ink-faint">
        Enabled skills can influence instructions when relevant. They do not grant browser, terminal, secret, or account permissions.
      </p>
    </div>
  );
}
