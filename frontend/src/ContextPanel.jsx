import React, { useEffect, useMemo, useState } from 'react';

const SOURCE_OPTIONS = [
  ['@context', 'All context'],
  ['@memory', 'Distilled memories'],
  ['@claude', 'Claude conversations'],
  ['@chatgpt', 'ChatGPT conversations'],
  ['@obsidian', 'Obsidian vault'],
  ['@theriad', 'Theriad project'],
  ['@kalshi', 'Kalshi project'],
];

const PROVIDERS = ['generic', 'claude', 'chatgpt', 'obsidian', 'gemini', 'perplexity', 'copilot', 'other'];
const MEMORY_KINDS = [
  'project_decision',
  'user_preference',
  'technical_note',
  'failed_attempt',
  'artifact_summary',
  'conversation_summary',
  'external_reference',
  'todo',
  'open_question',
];

async function jsonRequest(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}

function splitCsv(value) {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function memoryProvider(memory) {
  return memory.source?.provider || memory.source?.type || 'manual';
}

export function ContextPanel() {
  const [query, setQuery] = useState('@context model routing');
  const [results, setResults] = useState([]);
  const [memories, setMemories] = useState([]);
  const [status, setStatus] = useState('Idle');
  const [error, setError] = useState('');

  const [importForm, setImportForm] = useState({
    provider: 'generic',
    title: '',
    sourceFileName: '',
    tags: '',
    text: '',
    jsonText: '',
  });

  const [saveForm, setSaveForm] = useState({
    kind: 'project_decision',
    project: 'theriad',
    title: '',
    summary: '',
    body: '',
    tags: '',
    entities: '',
  });

  const availability = useMemo(() => {
    const counts = new Map();
    for (const memory of memories) {
      const provider = memoryProvider(memory);
      counts.set(provider, (counts.get(provider) || 0) + 1);
      if (memory.project) {
        counts.set(memory.project, (counts.get(memory.project) || 0) + 1);
      }
      counts.set('all', (counts.get('all') || 0) + 1);
    }
    return counts;
  }, [memories]);

  useEffect(() => {
    refreshMemories();
  }, []);

  async function refreshMemories() {
    try {
      const response = await fetch('/api/context/memories?status=active&limit=100');
      const data = await response.json();
      setMemories(data.memories || []);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function searchContext(event) {
    event?.preventDefault();
    setError('');
    setStatus('Searching context…');
    try {
      const data = await jsonRequest('/api/context/parse-command', { message: query });
      setResults(data.results || []);
      setStatus(data.final_answer || `Found ${(data.results || []).length} memory cards`);
    } catch (requestError) {
      setError(requestError.message);
      setStatus('Search failed');
    }
  }

  async function importContext(event) {
    event.preventDefault();
    setError('');
    setStatus('Importing context…');
    try {
      const body = {
        provider: importForm.provider,
        title: importForm.title || undefined,
        sourceFileName: importForm.sourceFileName || undefined,
        tags: splitCsv(importForm.tags),
        text: importForm.text || undefined,
      };
      if (importForm.provider === 'chatgpt' && importForm.jsonText.trim()) {
        body.jsonData = JSON.parse(importForm.jsonText);
        delete body.text;
      }
      const data = await jsonRequest('/api/context/import', body);
      setStatus(
        `Imported ${data.importedConversations} conversation(s), ${data.importedMessages} message(s), and created ${data.memoryCardsCreated} memory card(s).`
      );
      setImportForm((current) => ({ ...current, text: '', jsonText: '' }));
      await refreshMemories();
    } catch (requestError) {
      setError(requestError.message);
      setStatus('Import failed');
    }
  }

  async function saveMemory(event) {
    event.preventDefault();
    setError('');
    setStatus('Saving memory…');
    try {
      const data = await jsonRequest('/api/context/save', {
        kind: saveForm.kind,
        project: saveForm.project || undefined,
        title: saveForm.title,
        summary: saveForm.summary,
        body: saveForm.body || undefined,
        tags: splitCsv(saveForm.tags),
        entities: splitCsv(saveForm.entities),
        source: { type: 'manual', provider: 'chat' },
      });
      setStatus(`Saved memory: ${data.memory.title}`);
      setSaveForm((current) => ({ ...current, title: '', summary: '', body: '' }));
      await refreshMemories();
    } catch (requestError) {
      setError(requestError.message);
      setStatus('Save failed');
    }
  }

  async function forgetMemory(memoryId) {
    setError('');
    setStatus('Archiving memory…');
    try {
      await jsonRequest('/api/context/forget', { memoryId });
      setStatus('Memory archived');
      setResults((current) => current.filter((result) => result.id !== memoryId));
      await refreshMemories();
    } catch (requestError) {
      setError(requestError.message);
      setStatus('Archive failed');
    }
  }

  async function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setImportForm((current) => ({
      ...current,
      sourceFileName: file.name,
      title: current.title || file.name.replace(/\.[^.]+$/, ''),
      text,
    }));
  }

  return (
    <section className="context-panel" data-testid="context-panel">
      <header>
        <h2>Context Library</h2>
        <p>Import, save, search, and recall distilled memory cards without adding all memories to every prompt.</p>
      </header>

      <div className="context-autocomplete" data-testid="context-source-options">
        {SOURCE_OPTIONS.map(([token, label]) => (
          <button key={token} type="button" onClick={() => setQuery(`${token} `)}>
            <code>{token}</code> {label}
            <small>
              {' '}
              {availability.get(token.replace('@context', 'all').replace('@openrouter', 'theriad').replace('@', '')) || 0}
            </small>
          </button>
        ))}
      </div>

      <p role="status" data-testid="context-status">{status}</p>
      {error ? <p role="alert" data-testid="context-error">{error}</p> : null}

      <form data-testid="context-import-form" onSubmit={importContext}>
        <h3>Import Context</h3>
        <label>
          Provider
          <select
            value={importForm.provider}
            onChange={(event) => setImportForm((current) => ({ ...current, provider: event.target.value }))}
          >
            {PROVIDERS.map((provider) => <option key={provider} value={provider}>{provider}</option>)}
          </select>
        </label>
        <label>
          Title
          <input
            value={importForm.title}
            onChange={(event) => setImportForm((current) => ({ ...current, title: event.target.value }))}
            placeholder="Theriad routing discussion"
          />
        </label>
        <label>
          Source file name
          <input
            value={importForm.sourceFileName}
            onChange={(event) => setImportForm((current) => ({ ...current, sourceFileName: event.target.value }))}
            placeholder="claude-routing.md"
          />
        </label>
        <label>
          Upload text/Markdown transcript
          <input type="file" accept=".txt,.md,.markdown,.json" onChange={loadFile} />
        </label>
        <label>
          Tags
          <input
            value={importForm.tags}
            onChange={(event) => setImportForm((current) => ({ ...current, tags: event.target.value }))}
            placeholder="theriad, routing"
          />
        </label>
        <label>
          Transcript or note text
          <textarea
            value={importForm.text}
            onChange={(event) => setImportForm((current) => ({ ...current, text: event.target.value }))}
            placeholder="Human:&#10;What did we decide?&#10;&#10;Claude:&#10;Use explicit provider-aware memory cards."
          />
        </label>
        {importForm.provider === 'chatgpt' ? (
          <label>
            ChatGPT JSON
            <textarea
              data-testid="context-chatgpt-json"
              value={importForm.jsonText}
              onChange={(event) => setImportForm((current) => ({ ...current, jsonText: event.target.value }))}
              placeholder='{"title":"Open Router Context","messages":[...]}'
            />
          </label>
        ) : null}
        <button type="submit">Import into Context</button>
      </form>

      <form data-testid="context-save-form" onSubmit={saveMemory}>
        <h3>Remember Manually</h3>
        <label>
          Kind
          <select
            value={saveForm.kind}
            onChange={(event) => setSaveForm((current) => ({ ...current, kind: event.target.value }))}
          >
            {MEMORY_KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
          </select>
        </label>
        <label>
          Project
          <input
            value={saveForm.project}
            onChange={(event) => setSaveForm((current) => ({ ...current, project: event.target.value }))}
            placeholder="theriad"
          />
        </label>
        <label>
          Title
          <input
            required
            value={saveForm.title}
            onChange={(event) => setSaveForm((current) => ({ ...current, title: event.target.value }))}
            placeholder="Use explicit context recall"
          />
        </label>
        <label>
          Summary
          <textarea
            required
            value={saveForm.summary}
            onChange={(event) => setSaveForm((current) => ({ ...current, summary: event.target.value }))}
            placeholder="Theriad should retrieve memory only when @source commands require it."
          />
        </label>
        <label>
          Body
          <textarea
            value={saveForm.body}
            onChange={(event) => setSaveForm((current) => ({ ...current, body: event.target.value }))}
            placeholder="Optional longer note or evidence."
          />
        </label>
        <label>
          Tags
          <input
            value={saveForm.tags}
            onChange={(event) => setSaveForm((current) => ({ ...current, tags: event.target.value }))}
            placeholder="context, routing"
          />
        </label>
        <label>
          Entities
          <input
            value={saveForm.entities}
            onChange={(event) => setSaveForm((current) => ({ ...current, entities: event.target.value }))}
            placeholder="START_URL, Chrome MCP"
          />
        </label>
        <button type="submit">Remember</button>
      </form>

      <form data-testid="context-search-form" onSubmit={searchContext}>
        <h3>Search / Recall</h3>
        <label>
          Search or recall command
          <input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <button type="submit">Search Context</button>
      </form>

      <section data-testid="context-results">
        <h3>Retrieved Context</h3>
        <MemoryList memories={results} onForget={forgetMemory} emptyText="No retrieved memories yet." />
      </section>

      <section data-testid="context-memory-browser">
        <header>
          <h3>Stored Memories</h3>
          <button type="button" onClick={refreshMemories}>Refresh</button>
        </header>
        <MemoryList memories={memories} onForget={forgetMemory} emptyText="No active memories stored yet." />
      </section>
    </section>
  );
}

function MemoryList({ memories, onForget, emptyText }) {
  if (!memories.length) {
    return <p>{emptyText}</p>;
  }

  return (
    <ol aria-label="Context memories">
      {memories.map((memory) => (
        <li key={memory.id} data-testid="context-memory-card">
          <strong>{memory.title}</strong>
          <p>{memory.summary}</p>
          <small>
            {memoryProvider(memory)} · {memory.kind}
            {memory.project ? ` · ${memory.project}` : ''}
          </small>
          {memory.tags?.length ? <p>Tags: {memory.tags.join(', ')}</p> : null}
          <button type="button" onClick={() => onForget(memory.id)}>Archive</button>
        </li>
      ))}
    </ol>
  );
}

