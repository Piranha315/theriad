/**
 * Chat.jsx — Chat-first workspace page.
 *
 * Layout: consolidated app rail | main chat panel.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import CostConfirmationDialog from './CostConfirmationDialog.jsx';
import { nextMusing, agentSynopsis } from './thinkingMuse.js';
import { toFriendlyError } from './uiErrors.js';
import {
  buildPreviewDoc,
  extractPreviewArtifact,
} from './previewArtifact.js';

// @mentions select named memory uploads returned by /api/memory/mentions.
// Keep a stable palette so the same upload remains visually identifiable while
// users edit its alias.
const MEMORY_COLORS = ['#f97316', '#3b82f6', '#22c55e', '#a855f7', '#14b8a6', '#eab308'];

const DEFAULT_RUNTIME_SETTINGS = {
  regular_chat_returns_answer: true,
  allow_follow_up_questions: true,
  route_only_available: false,
  model_origin_policy: 'any',
};

function tokenizePrompt(text, memoryMentions = []) {
  const memoryColors = Object.fromEntries(
    memoryMentions.map((mention, index) => [mention.alias, MEMORY_COLORS[index % MEMORY_COLORS.length]])
  );
  const tokens = [];
  const mentionPattern = /@([A-Za-z0-9_-]+)/g;
  let lastIndex = 0;
  let match;

  while ((match = mentionPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ text: text.slice(lastIndex, match.index), color: null });
    }
    tokens.push({
      text: match[0],
      color: memoryColors[match[1].toLowerCase()] ?? null,
    });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    tokens.push({ text: text.slice(lastIndex), color: null });
  }
  return tokens;
}

// ── API helpers ───────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    ...rest,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    const method = (rest.method ?? 'GET').toUpperCase();
    const msg = `API ${res.status} on ${method} ${path}: ${body}`;
    // Guest workspaces intentionally receive 401s from private persistence
    // routes. They are an expected empty-state boundary, not a frontend
    // failure worth emitting as a console error.
    if (res.status !== 401) {
      console.error('[Chat] API error', { method, path, status: res.status, body });
    }
    const error = new Error(msg);
    error.status = res.status;
    try {
      const parsed = JSON.parse(body);
      error.serverMessage = parsed?.message || parsed?.error || parsed?.detail || null;
    } catch { /* non-JSON error body */ }
    throw error;
  }
  return res.json();
}

// Compact summary of what the model did on the web for one answer.
function webActivityLabel(web) {
  const steps = Array.isArray(web?.steps) ? web.steps : [];
  if (!steps.length) return '';
  const searches = steps
    .filter(step => step.type === 'search')
    .reduce((total, step) => total + (Number(step.count) || 1), 0);
  const hosts = [...new Set(
    steps
      .filter(step => (step.type === 'fetch' || step.type === 'browse') && step.url)
      .map(step => {
        try { return new URL(step.url).hostname.replace(/^www\./, ''); } catch { return null; }
      })
      .filter(Boolean),
  )];
  const parts = [];
  if (searches) parts.push(searches === 1 ? 'searched the web' : `searched the web ${searches}×`);
  if (hosts.length) parts.push(`read ${hosts.slice(0, 3).join(', ')}${hosts.length > 3 ? '…' : ''}`);
  return parts.join(' · ');
}

// ── Markdown message rendering ───────────────────────────────────────────────

function renderInlineMarkdown(text, keyPrefix) {
  const tokenPattern = /(`[^`\n]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\)|\*\*[^*\n]+?\*\*|__[^_\n]+?__|(?<!\*)\*[^*\n]+?\*(?!\*)|(?<!_)_[^_\n]+?_(?!_))/g;
  const parts = [];
  let lastIndex = 0;
  let match;
  let tokenIndex = 0;

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));

    const token = match[0];
    const key = `${keyPrefix}-${tokenIndex++}`;
    if (token.startsWith('`')) {
      parts.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith('[')) {
      const linkMatch = token.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
      parts.push(
        <a key={key} href={linkMatch[2]} target="_blank" rel="noreferrer">
          {linkMatch[1]}
        </a>
      );
    } else if (token.startsWith('**') || token.startsWith('__')) {
      parts.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    lastIndex = tokenPattern.lastIndex;
  }

  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

async function copyText(text) {
  const value = String(text ?? '');
  if (!value) return false;

  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return true;
  }

  const textarea = document.createElement('textarea');
  textarea.value = value;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand('copy');
  textarea.remove();
  return copied;
}

function formatMessageTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function shortModelName(value) {
  if (!value) return 'Best fit';
  const name = String(value).split('/').pop().replace(/[-_]/g, ' ').trim();
  const haiku = name.match(/claude\s+([\d.]+)\s+haiku/i);
  if (haiku) return `Haiku ${haiku[1]}`;
  const titleCased = name.replace(/\b\w/g, character => character.toUpperCase());
  return titleCased.length > 24 ? `${titleCased.slice(0, 22)}…` : titleCased;
}

function formatHeaderCost(value) {
  const cost = Number(value);
  if (!Number.isFinite(cost) || cost <= 0) return '$0.00';
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

function MarkdownCodeBlock({ language, value }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      const didCopy = await copyText(value);
      if (!didCopy) return;
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // Copy is a convenience; rendering the answer must never depend on it.
    }
  }

  return (
    <div className="theriad-code-block">
      <div className="theriad-code-block__header">
        <span>{language || 'code'}</span>
        <button type="button" onClick={handleCopy} aria-label="Copy code" title="Copy code">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>
        <code className={language ? `language-${language}` : undefined}>{value}</code>
      </pre>
    </div>
  );
}

function MarkdownContent({ content }) {
  const lines = String(content ?? '').replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let paragraph = [];
  let list = null;
  let code = null;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const text = paragraph.join(' ').trim();
    if (text) {
      const blockIndex = blocks.length;
      blocks.push(
        <p key={`paragraph-${blockIndex}`}>
          {renderInlineMarkdown(text, `paragraph-${blockIndex}`)}
        </p>
      );
    }
    paragraph = [];
  };

  const flushList = () => {
    if (!list) return;
    const ListTag = list.type === 'ordered' ? 'ol' : 'ul';
    const blockIndex = blocks.length;
    blocks.push(
      <ListTag key={`list-${blockIndex}`}>
        {list.items.map((item, index) => (
          <li key={`list-item-${index}`}>
            {renderInlineMarkdown(item, `list-${blockIndex}-${index}`)}
          </li>
        ))}
      </ListTag>
    );
    list = null;
  };

  const flushCode = () => {
    if (!code) return;
    blocks.push(
      <MarkdownCodeBlock
        key={`code-${blocks.length}`}
        language={code.language}
        value={code.lines.join('\n')}
      />
    );
    code = null;
  };

  lines.forEach(line => {
    const fence = line.match(/^\s*```\s*([\w-]*)\s*$/);
    if (code) {
      if (fence) flushCode();
      else code.lines.push(line);
      return;
    }
    if (fence) {
      flushParagraph();
      flushList();
      code = { language: fence[1], lines: [] };
      return;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      return;
    }

    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      flushParagraph();
      flushList();
      const HeadingTag = `h${heading[1].length}`;
      const blockIndex = blocks.length;
      blocks.push(
        <HeadingTag key={`heading-${blockIndex}`}>
          {renderInlineMarkdown(heading[2], `heading-${blockIndex}`)}
        </HeadingTag>
      );
      return;
    }

    const listItem = line.match(/^\s*([-*+] |\d+[.)] )(.+)$/);
    if (listItem) {
      flushParagraph();
      const type = /^\d/.test(listItem[1]) ? 'ordered' : 'unordered';
      if (list && list.type !== type) flushList();
      if (!list) list = { type, items: [] };
      list.items.push(listItem[2]);
      return;
    }

    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      flushList();
      const blockIndex = blocks.length;
      blocks.push(
        <blockquote key={`quote-${blockIndex}`}>
          {renderInlineMarkdown(quote[1], `quote-${blockIndex}`)}
        </blockquote>
      );
      return;
    }

    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      flushParagraph();
      flushList();
      blocks.push(<hr key={`rule-${blocks.length}`} />);
      return;
    }

    flushList();
    paragraph.push(line.trim());
  });

  flushParagraph();
  flushList();
  flushCode();

  return <div className="markdown-content">{blocks}</div>;
}

// ── Utility helpers ───────────────────────────────────────────────────────────

// Images are routed to a vision-capable model server-side; everything else is
// text-extracted. Keep in sync with ALLOWED_EXTENSIONS in theriad/api/projects.py
// — the server enforces this list, this copy only makes rejection immediate.
const IMAGE_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp',
]);

const TEXT_EXTENSIONS = new Set([
  '.txt', '.md', '.json', '.csv',
  '.py', '.js', '.jsx', '.ts', '.tsx',
  '.html', '.css', '.sql', '.yaml', '.yml',
]);

const ALLOWED_EXTENSIONS = new Set([...TEXT_EXTENSIONS, ...IMAGE_EXTENSIONS]);

// Matches MAX_IMAGE_BYTES server-side. Rejecting here saves a round trip on a
// file the server would refuse anyway.
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

function isImageFile(filename) {
  return IMAGE_EXTENSIONS.has(extOf(filename || ''));
}

function extOf(filename) {
  const idx = filename.lastIndexOf('.');
  return idx >= 0 ? filename.slice(idx).toLowerCase() : '';
}

function fmtCost(usd) {
  if (usd == null) return '—';
  return `$${usd.toFixed(6)}`;
}

function friendlyCost(usd) {
  if (usd == null) return 'Not available';
  if (usd === 0) return 'Free';
  // Adaptive precision: keep tiny costs informative instead of "Under 1¢".
  if (usd < 0.0001) return `$${usd.toFixed(6)}`;
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `${(usd * 100).toFixed(usd < 0.10 ? 1 : 0)}¢`;
  return `$${usd.toFixed(2)}`;
}

function fmtPerMtok(usd) {
  if (usd == null) return null;
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
}

// ── Micro-components ──────────────────────────────────────────────────────────

const Spinner = ({ size = 'sm' }) => (
  <svg
    className={`animate-spin ${size === 'sm' ? 'w-4 h-4' : 'w-5 h-5'} text-accent`}
    viewBox="0 0 24 24" fill="none"
  >
    <circle className="opacity-25" cx="12" cy="12" r="10"
      stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
  </svg>
);

// A transparent, reusable signal for the thinking state. Keeping the path in
// the DOM lets the surrounding surface show through instead of baking a
// rectangular background into a media asset.
const THINKING_SIGNAL_PATH = 'M2 16 C7 16 8 9 12 9 S17 23 21 23 S25 7 29 7 S34 20 38 20 S41 16 46 16';

const ThinkingSignal = () => (
  <svg
    className="theriad-thinking-signal"
    viewBox="0 0 48 32"
    fill="none"
    focusable="false"
    aria-hidden="true"
  >
    <path className="theriad-thinking-signal__track" d={THINKING_SIGNAL_PATH} />
    <path
      className="theriad-thinking-signal__pulse"
      d={THINKING_SIGNAL_PATH}
      pathLength="1"
    />
    <circle className="theriad-thinking-signal__node" cx="46" cy="16" r="1.35" />
  </svg>
);

// How long the model must go without a fresh status update before the
// indicator starts musing (rotating quotes/tips), and how long each musing
// dwells on screen before the next one fades in.
const THINKING_IDLE_DELAY_MS = 4000;
const THINKING_MUSE_DWELL_MS = 7000;

// Thinking indicator. Shows the live synopsis (`status`) whenever the model has
// something specific to report — searching, reading a page, running a command.
// When the status goes quiet (genuine thinking, no update for a few seconds) it
// crossfades through a shuffled mix of quotes and tips. A new live status
// always interrupts and takes precedence.
const TheriadLoadingAnimation = ({ status, label }) => {
  const live = ((status ?? label) ?? '').trim();
  const [muse, setMuse] = useState(null);
  const deckRef = useRef({ current: null, last: null });

  // A change in live status resets everything: show it immediately, then start
  // an idle timer that begins musing only if no newer status arrives.
  useEffect(() => {
    setMuse(null);
    const firstDelay = live ? THINKING_IDLE_DELAY_MS : 600;
    const t = setTimeout(() => setMuse(nextMusing(deckRef.current)), firstDelay);
    return () => clearTimeout(t);
  }, [live]);

  // While musing, rotate to the next quote/tip on a steady cadence.
  const musing = muse != null;
  useEffect(() => {
    if (!musing) return undefined;
    const t = setInterval(() => setMuse(nextMusing(deckRef.current)), THINKING_MUSE_DWELL_MS);
    return () => clearInterval(t);
  }, [musing]);

  return (
    <div className="flex items-center gap-3 min-w-0" role="status" aria-live="polite">
      <ThinkingSignal />
      {muse ? (
        // Musings are decorative: keep them out of the live region so screen
        // readers hear a single stable "thinking" note, not every quote.
        <div className="min-w-0" aria-hidden="true">
          <div key={muse.text} className="theriad-thinking-line">
            {muse.kind === 'quote' ? (
              <>
                <p className="text-xs text-ink-muted leading-relaxed">“{muse.text}”</p>
                <p className="text-[10px] text-ink-faint mt-0.5">{muse.author}</p>
              </>
            ) : (
              <p className="text-xs text-ink-muted leading-relaxed">
                <span className="text-accent font-medium">Tip</span>
                <span className="text-ink-faint"> · </span>
                {muse.text}
              </p>
            )}
          </div>
          <span className="sr-only">Theriad is thinking…</span>
        </div>
      ) : (
        <span key={live} className="theriad-thinking-line text-xs text-ink-muted truncate">
          {live || 'Thinking…'}
        </span>
      )}
    </div>
  );
};

const TierBadge = ({ tier }) => {
  const cls = {
    free_or_floor:    'bg-ok/10 text-ok border-ok/30',
    cheap_bulk:       'bg-ok/10 text-ok border-ok/30',
    cheap_code:       'bg-ok/10 text-ok border-ok/30',
    code_strong:      'bg-accent/10    text-accent    border-accent/30',
    strong_reasoning: 'bg-accent/10    text-accent    border-accent/30',
    premium:          'bg-warn/10   text-warn   border-warn/30',
  }[tier] ?? 'bg-ink-faint/10 text-ink-muted border-line-strong';
  return (
    <span className={`px-1.5 py-0.5 text-xs rounded border font-mono ${cls}`}>{tier}</span>
  );
};

// ── Model vendor logos ───────────────────────────────────────────────────────

const MODEL_VENDORS = [
  { match: ['openai', 'gpt', 'o1', 'o3', 'o4'], name: 'OpenAI',    color: '#10a37f', glyph: 'hex',   logo: '/models/gpt.svg' },
  { match: ['anthropic', 'claude'],             name: 'Anthropic', color: '#d97757', glyph: 'A',     logo: '/models/opus.svg' },
  { match: ['google', 'gemini', 'gemma'],       name: 'Google',    color: '#4f8af5', glyph: 'spark' },
  { match: ['meta', 'llama'],                   name: 'Meta',      color: '#0668e1', glyph: 'M',     logo: '/models/meta.png' },
  { match: ['mistral', 'mixtral', 'ministral'], name: 'Mistral',   color: '#fa5f1c', glyph: 'M',     logo: '/models/mistral.png' },
  { match: ['qwen'],                            name: 'Qwen',      color: '#8b5cf6', glyph: 'Q',     logo: '/models/qwen.png' },
  { match: ['deepseek'],                        name: 'DeepSeek',  color: '#4d6bfe', glyph: 'D',     logo: '/models/deepseek.svg' },
  { match: ['nvidia', 'nemotron'],              name: 'NVIDIA',    color: '#76b900', glyph: 'N',     logo: '/models/nemo.svg' },
  { match: ['xiaomi', 'mimo'],                  name: 'Xiaomi',    color: '#ff6900', glyph: 'Mi',    logo: '/models/mimo.svg' },
  { match: ['x-ai', 'grok'],                    name: 'xAI',       color: '#e8e8e8', glyph: 'X',     logo: '/models/grok.png' },
  { match: ['ollama'],                          name: 'Ollama',    color: '#c9d4ea', glyph: 'O' },
  { match: ['moonshot', 'kimi'],                name: 'Moonshot',  color: '#38bdf8', glyph: 'K',     logo: '/models/kimi.png' },
  { match: ['z-ai', 'glm', 'zhipu'],            name: 'Z.ai',      color: '#34d399', glyph: 'Z',     logo: '/models/zai.png' },
  { match: ['poolside', 'laguna'],              name: 'Poolside',  color: '#8b9cff', glyph: 'L',     logo: '/models/laguna.png' },
];

function vendorForModel(model) {
  const slug = String(model || '').toLowerCase();
  return MODEL_VENDORS.find(v => v.match.some(m => slug.includes(m))) ?? null;
}

function ModelLogo({ model, size = 'small' }) {
  const vendor = vendorForModel(model);
  const sizeClass = size === 'avatar' ? 'theriad-model-logo--avatar' : 'theriad-model-logo--small';
  if (!vendor) {
    return (
      <span
        title="Model provider"
        className={`theriad-model-logo ${sizeClass}`}
      >
        AI
      </span>
    );
  }
  const { name, color, glyph } = vendor;
  return (
    <span
      title={name}
      aria-label={`${name} logo`}
      className={`theriad-model-logo ${sizeClass}`}
      style={{ backgroundColor: `${color}22`, border: `1px solid ${color}55` }}
    >
      {vendor.logo ? (
        <img src={vendor.logo} alt="" aria-hidden="true" />
      ) : glyph === 'hex' ? (
        <svg viewBox="0 0 24 24" className="w-2.5 h-2.5" fill="none"
          stroke={color} strokeWidth="2.4" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 2.5 20.2 7.25v9.5L12 21.5 3.8 16.75v-9.5z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      ) : glyph === 'spark' ? (
        <svg viewBox="0 0 24 24" className="w-2.5 h-2.5" fill={color} aria-hidden="true">
          <path d="M12 1.5c.6 5.8 4.7 9.9 10.5 10.5-5.8.6-9.9 4.7-10.5 10.5C11.4 16.7 7.3 12.6 1.5 12 7.3 11.4 11.4 7.3 12 1.5z" />
        </svg>
      ) : (
        <span className="text-[9px] font-bold leading-none" style={{ color }}>{glyph}</span>
      )}
    </span>
  );
}

// ── Dropzone wrapper ──────────────────────────────────────────────────────────

function Dropzone({ projectId, onUploaded, disabled, children }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dropError, setDropError] = useState(null);
  const counter = useRef(0);

  function onDragEnter(e) {
    e.preventDefault();
    counter.current += 1;
    setDragging(true);
  }
  function onDragLeave(e) {
    e.preventDefault();
    counter.current -= 1;
    if (counter.current === 0) setDragging(false);
  }
  function onDragOver(e) { e.preventDefault(); }

  async function onDrop(e) {
    e.preventDefault();
    counter.current = 0;
    setDragging(false);
    setDropError(null);

    if (disabled) return;

    const files = [...e.dataTransfer.files];
    const problems = [];
    const accepted = [];
    for (const f of files) {
      if (!ALLOWED_EXTENSIONS.has(extOf(f.name))) {
        problems.push(`Unsupported: ${f.name}`);
      } else if (isImageFile(f.name) && f.size > MAX_IMAGE_BYTES) {
        problems.push(
          `Too large: ${f.name} (${(f.size / 1048576).toFixed(1)} MB, max ${MAX_IMAGE_BYTES / 1048576} MB)`
        );
      } else {
        accepted.push(f);
      }
    }
    if (problems.length) setDropError(problems.join(' · '));
    if (!accepted.length) return;

    setUploading(true);
    try {
      for (const file of accepted) {
        const fd = new FormData();
        fd.append('file', file);
        const uploadPath = projectId
          ? `/api/projects/${projectId}/files`
          : '/api/files';
        const res = await fetch(uploadPath, { method: 'POST', body: fd });
        if (!res.ok) {
          const body = await res.text().catch(() => res.statusText);
          const uploadError = new Error(`API ${res.status} on POST ${uploadPath}: ${body}`);
          uploadError.status = res.status;
          throw uploadError;
        }
        const saved = await res.json();
        onUploaded(saved);
      }
    } catch (err) {
      setDropError(err.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
      className="relative flex-1 min-w-0"
    >
      {dragging && (
        <div className="absolute inset-0 z-20 rounded-xl bg-accent/10 border-2 border-accent border-dashed flex items-center justify-center pointer-events-none">
          <p className="text-accent font-medium text-sm">Drop to attach</p>
        </div>
      )}
      {uploading && (
        <div className="absolute top-2 right-3 z-20 flex items-center gap-1.5 text-xs text-accent">
          <Spinner /> Uploading…
        </div>
      )}
      {dropError && (
        <div className="absolute bottom-2 left-3 right-3 z-20 text-xs text-warn bg-warn/10 border border-warn/20 rounded px-2 py-1">
          {dropError}
          <button className="ml-2 underline" onClick={() => setDropError(null)}>✕</button>
        </div>
      )}
      {children}
    </div>
  );
}

// ── WebPreviewPanel ───────────────────────────────────────────────────────────

function WebPreviewPanel({ srcdoc, onClose }) {
  return (
    <div className="border-t border-line mt-3 pt-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-ink-muted font-medium">Preview</span>
        <button
          onClick={onClose}
          className="text-xs text-ink-faint hover:text-ink"
        >✕</button>
      </div>
      <iframe
        srcDoc={srcdoc}
        sandbox="allow-scripts"
        className="w-full rounded border border-line bg-white"
        style={{ height: 260, colorScheme: 'light' }}
        title="output preview"
      />
    </div>
  );
}

function PreviewInspector({ preview, onClose }) {
  const artifact = preview?.artifact;
  const legacyPreview = preview?.preview;
  const [tailwindEnabled, setTailwindEnabled] = useState(
    artifact?.kind === 'tailwind_html' || /cdn\.tailwindcss\.com/i.test(artifact?.html ?? '')
  );

  if (!artifact && !legacyPreview) return null;

  const srcDoc = artifact
    ? buildPreviewDoc(artifact, tailwindEnabled)
    : legacyPreview.srcdoc;
  // allow-scripts unconditionally: the Tailwind checkbox is a styling choice,
  // not a JavaScript switch. Never add allow-same-origin here — combined with
  // allow-scripts it would let the frame escape its own sandbox.
  const sandbox = 'allow-scripts';

  return (
    <aside className="theriad-preview-inspector" aria-label="Preview inspector">
      <div className="theriad-preview-inspector__header">
        <div className="min-w-0">
          <p className="theriad-preview-inspector__eyebrow">Inspector</p>
          <h2 className="theriad-preview-inspector__title">Preview</h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close preview inspector"
          className="theriad-preview-inspector__close"
        >
          ✕
        </button>
      </div>

      <div className="theriad-preview-inspector__meta">
        <span>Live artifact</span>
        {artifact?.kind && <span className="font-mono">{artifact.kind}</span>}
      </div>

      <div className="theriad-preview-inspector__body">
        {artifact?.warnings.map((warning, index) => (
          <p key={index} className="px-4 pt-3 text-xs text-warn">⚠ {warning}</p>
        ))}
        <div className="theriad-preview-inspector__canvas">
          <iframe
            key={`${preview?.sourceKey}_${tailwindEnabled}`}
            srcDoc={srcDoc}
            sandbox={sandbox}
            title="Preview inspector"
          />
        </div>
      </div>

      <div className="theriad-preview-inspector__footer">
        {artifact && (
          <label className="flex items-center gap-2 text-xs text-ink-muted cursor-pointer select-none">
            <input
              type="checkbox"
              checked={tailwindEnabled}
              onChange={event => setTailwindEnabled(event.target.checked)}
              className="accent-accent"
            />
            Tailwind CDN
          </label>
        )}
        <p>Sandboxed preview · external requests blocked</p>
      </div>
    </aside>
  );
}

// ── Inline route details ──────────────────────────────────────────────────────

function RouteDetails({ route, run, context, preview, capability, usage, onPreviewClose }) {
  const [advanced, setAdvanced] = useState(false);
  if (!route && !capability) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-ink-faint text-xs text-center px-4">
        <p>No routing details are available for this response.</p>
      </div>
    );
  }

  const cost = run?.actual_cost_usd ?? route?.estimated_cost_usd ?? 0;
  const usedBrowser = route?.intent === 'browser_agent' ||
    capability?.recommended_execution_mode === 'browser_agent';
  const isPremium = route?.tier === 'premium';
  const simpleSummary = usedBrowser
    ? 'Answered using live web tools and a cost-efficient model.'
    : isPremium
      ? 'Answered using a higher-quality model because the task needed it.'
      : 'Answered using a fast, low-cost model selected for this request.';

  if (!advanced) {
    return (
      <div className="p-4 overflow-y-auto text-xs">
        <div className="rounded-xl bg-surface-hover border border-line px-4 py-4">
          <p className="text-sm text-ink leading-relaxed">{simpleSummary}</p>
          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-ink-muted">{run ? 'Final cost' : 'Estimated cost'}</span>
            <span className="text-ok">{friendlyCost(cost)}</span>
          </div>
          {run?.prompt_cache_used && (
            <div className="mt-1.5 flex items-center justify-between gap-3">
              <span className="text-ink-muted">Prompt reused</span>
              <span className="text-ok">Yes</span>
            </div>
          )}
          {typeof route?.savings_pct === 'number' && route.savings_pct > 0 && (
            <div className="mt-1.5 flex items-center justify-between gap-3">
              <span className="text-ink-muted">Estimated savings</span>
              <span className="font-mono text-ok">{route.savings_pct}%</span>
            </div>
          )}
          {usage?.has_data && usage.blended_cost_per_mtok != null && (
            <div className="mt-3 border-t border-line pt-2.5">
              <div className="flex items-center justify-between gap-3">
                <span className="text-ink-muted">Your projected rate</span>
                <span className="font-mono text-accent">{fmtPerMtok(usage.blended_cost_per_mtok)}/1M tok</span>
              </div>
              {usage.baseline_cost_per_mtok != null && (
                <p className="mt-1 text-[10px] text-ink-faint leading-snug">
                  Blend of every model you've used so far · same tokens on {usage.baseline_model} would run {fmtPerMtok(usage.baseline_cost_per_mtok)}/1M
                </p>
              )}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={() => setAdvanced(true)}
          className="mt-3 text-xs text-ink-faint hover:text-ink transition-colors"
        >
          Show advanced details
        </button>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-5 overflow-y-auto text-xs">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium text-ink">Advanced details</p>
        <button
          type="button"
          onClick={() => setAdvanced(false)}
          className="text-xs text-ink-faint hover:text-ink transition-colors"
        >
          Simple view
        </button>
      </div>
      {/* Context Layer */}
      {route && (
        <section>
          <p className="text-ink-faint uppercase tracking-widest text-[10px] mb-2">Context Layer</p>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between gap-3">
              <span className="text-ink-muted">Intent</span>
              <span className="font-mono text-accent text-right truncate max-w-[130px]">
                {route.intent ?? '—'}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-ink-muted">Model</span>
              <span className="font-mono text-ink text-right truncate max-w-[130px]">
                {route.model ?? '—'}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-ink-muted">Policy</span>
              <span className="font-mono text-ink text-right truncate max-w-[130px]">
                {route.policy ?? '—'}
              </span>
            </div>
            {route.decision_source && (
              <div className="flex items-center justify-between gap-3">
                <span className="text-ink-muted">Classified by</span>
                <span className="font-mono text-ok text-right">{route.decision_source}</span>
              </div>
            )}
            {typeof route.browser_confidence === 'number' && (
              <div className="flex items-center justify-between gap-3">
                <span className="text-ink-muted">Browser confidence</span>
                <span className="font-mono text-ink">
                  {(route.browser_confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>

          {capability?.capabilities && (
            <div className="mt-3 border-t border-line pt-2">
              <p className="text-[10px] text-ink-faint mb-1.5">Capability flags</p>
              <div className="flex flex-wrap gap-1">
                {Object.entries(capability.capabilities).filter(([, active]) => active).length === 0 ? (
                  <p className="text-[11px] text-ink-faint italic">No capability flags fired.</p>
                ) : Object.entries(capability.capabilities)
                  .filter(([, active]) => active)
                  .map(([flag]) => (
                    <span
                      key={flag}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-surface-hover border border-line text-ink-faint"
                    >
                      {CAP_LABELS[flag] ?? flag}
                    </span>
                  ))}
              </div>
            </div>
          )}

          <div className="mt-3 border-t border-line pt-2">
            <p className="text-[10px] text-ink-faint mb-1.5">Memory nodes</p>
            {(route.memory_nodes ?? []).length === 0 ? (
              <p className="text-[11px] text-ink-faint italic">No memory nodes for this session.</p>
            ) : (
              <ul className="space-y-1">
                {route.memory_nodes.map((node, i) => (
                  <li key={i} className="text-[11px] text-ink-muted flex items-start gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0 mt-1" />
                    {node.label ?? node}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}

      {/* Route */}
      {route && <section>
        <p className="text-ink-faint uppercase tracking-widest text-[10px] mb-2">Route</p>
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-ink-muted">Tier</span>
            <TierBadge tier={route.tier ?? 'stored run'} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-ink-muted">Model</span>
            <span className="font-mono text-ink text-right max-w-[120px] truncate">{route.model}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-ink-muted">Confidence</span>
            <span className="text-ink">
              {typeof route.confidence === 'number' ? `${(route.confidence * 100).toFixed(0)}%` : 'Not stored'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-ink-muted">Est. cost</span>
            <span className="font-mono text-ok">{fmtCost(route.estimated_cost_usd)}</span>
          </div>
          {route.savings_pct > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">vs baseline</span>
              <span className="font-mono text-ok">−{route.savings_pct}%</span>
            </div>
          )}
          {route.routing_latency_ms != null && (
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Routing</span>
              <span className="text-ink">{route.routing_latency_ms.toFixed(0)}ms</span>
            </div>
          )}
        </div>
        {route.reason && (
          <p className="mt-2 text-ink-faint leading-relaxed">{route.reason}</p>
        )}
      </section>}

      {/* Context */}
      {context && (
        <section>
          <p className="text-ink-faint uppercase tracking-widest text-[10px] mb-2">Prompt Context</p>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Tokens</span>
              <span className="text-ink">{context.tokens ?? 0}</span>
            </div>
            {context.truncated && (
              <p className="text-warn">⚠ Context truncated</p>
            )}
            {context.history_summarized && (
              <p className="text-ok">
                Older chat condensed; {context.recent_messages_kept ?? 0} recent messages kept intact
              </p>
            )}
            {context.conversation_sources?.length > 0 && (
              <p className="text-ink-muted">
                Retrieved {context.conversation_sources.length} named conversation{context.conversation_sources.length === 1 ? '' : 's'} from Memory
              </p>
            )}
            {context.paid_warning && (
              <p className="text-warn">⚠ Paid provider warning</p>
            )}
          </div>
        </section>
      )}

      {/* Run */}
      {run && (
        <section>
          <p className="text-ink-faint uppercase tracking-widest text-[10px] mb-2">Run</p>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Actual cost</span>
              <span className="font-mono text-ok">{fmtCost(run.actual_cost_usd)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Latency</span>
              <span className="text-ink">{run.total_latency_ms?.toFixed(0)}ms</span>
            </div>
            {run.prompt_cache_used && (
              <div className="flex items-center justify-between">
                <span className="text-ink-muted">Prompt cache</span>
                <span className="text-ok">Reused</span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Tokens in/out</span>
              <span className="font-mono text-ink">{run.input_tokens}/{run.output_tokens}</span>
            </div>
            {run.fallback_used && (
              <p className="text-warn">⚠ Fallback used</p>
            )}
          </div>
        </section>
      )}

      {/* Capability */}
      {capability && (
        <section>
          <p className="text-ink-faint uppercase tracking-widest text-[10px] mb-2">Capability</p>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Mode</span>
              <span className="font-mono text-ink text-right text-[11px] truncate max-w-[120px]">
                {capability.recommended_execution_mode}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Risk</span>
              <span className={`font-mono text-[11px] ${
                capability.risk_level === 'high' ? 'text-danger'
                : capability.risk_level === 'medium' ? 'text-warn'
                : 'text-ok'
              }`}>{capability.risk_level}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">Confidence</span>
              <span className="text-ink">{(capability.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
          {capability.reason && (
            <p className="mt-2 text-ink-faint leading-relaxed">{capability.reason}</p>
          )}
          {capability.matched_signals?.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {capability.matched_signals.slice(0, 6).map((s, i) => (
                <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-surface-hover border border-line text-ink-faint font-mono">
                  {s}
                </span>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Preview */}
      {preview && (
        <WebPreviewPanel srcdoc={preview.srcdoc} onClose={onPreviewClose} />
      )}
    </div>
  );
}

// ── Capability recommendation card ───────────────────────────────────────────

const MODE_META = {
  browser_agent:    { label: 'Browser Agent',    color: 'text-info', bg: 'bg-info/10 border-info/30' },
  local_tool_agent: { label: 'Local Tool Agent', color: 'text-warn',  bg: 'bg-warn/10  border-warn/30'  },
  workflow_agent:   { label: 'Workflow Agent',   color: 'text-info', bg: 'bg-info/10 border-info/30' },
  hybrid:           { label: 'Hybrid Agent',     color: 'text-accent',   bg: 'bg-accent/10   border-accent/30'   },
  text_plus_preview:{ label: 'Text + Preview',   color: 'text-ok',bg: 'bg-ok/10 border-ok/30' },
  text_only:        { label: 'Text Only',        color: 'text-ink',  bg: 'bg-ink-faint/10  border-line-strong'  },
};

const RISK_META = {
  low:    { label: 'Low',    color: 'text-ok', dot: 'bg-ok' },
  medium: { label: 'Medium', color: 'text-warn',   dot: 'bg-warn'   },
  high:   { label: 'High',   color: 'text-danger',     dot: 'bg-danger'     },
};

const CAP_LABELS = {
  requires_current_web:        'Current web data',
  requires_browser_navigation: 'Browser navigation',
  requires_visual_observation: 'Visual observation',
  requires_dom_observation:    'DOM inspection',
  requires_file_editing:       'File editing',
  requires_local_command:      'Local commands',
  requires_code_execution:     'Code execution',
  requires_artifact_preview:   'Artifact preview',
  requires_form_interaction:   'Form interaction',
  requires_account_access:     'Account access',
  requires_sensitive_info:     'Sensitive info',
  requires_data_extraction:    'Data extraction',
  requires_user_approval:      'User approval required',
  requires_paid_call:          'Paid API call',
};

function CapabilityRecommendationCard({ capability, onRunAgent, onTextOnly, onCancel }) {
  const mode = capability.recommended_execution_mode;
  const risk = capability.risk_level;
  const modeMeta = MODE_META[mode] ?? MODE_META.text_only;
  const riskMeta = RISK_META[risk] ?? RISK_META.low;
  const isHighRisk = risk === 'high';
  const activeFlags = Object.entries(capability.capabilities)
    .filter(([, v]) => v)
    .map(([k]) => CAP_LABELS[k] ?? k);

  return (
    <div className={`mx-0 mb-4 rounded-xl border overflow-hidden min-w-0 max-w-full ${modeMeta.bg}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-line flex items-center gap-3 flex-wrap">
        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${riskMeta.dot}`} />
        <span className="text-xs text-ink-faint uppercase tracking-widest">Theriad recommends</span>
        <span className={`text-sm font-semibold ${modeMeta.color}`}>{modeMeta.label}</span>
        {isHighRisk && (
          <span className="text-xs text-danger bg-danger/10 border border-danger/30 rounded px-1.5 py-0.5">
            Approval required
          </span>
        )}
      </div>

      {/* Body */}
      <div className="px-4 py-3 space-y-3">
        {/* Reason */}
        <p className="text-xs text-ink leading-relaxed">{capability.reason}</p>

        {/* Active flags */}
        {activeFlags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {activeFlags.map(f => (
              <span
                key={f}
                className="text-[11px] px-2 py-0.5 rounded bg-surface-hover border border-line text-ink-muted"
              >
                {f}
              </span>
            ))}
          </div>
        )}

        {/* Risk + confidence */}
        <div className="flex items-center gap-4 text-xs text-ink-faint">
          <span>
            Risk: <span className={`font-medium ${riskMeta.color}`}>{riskMeta.label}</span>
          </span>
          <span>
            Confidence: <span className="font-mono text-ink-muted">
              {(capability.confidence * 100).toFixed(0)}%
            </span>
          </span>
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2 pt-1">
          {!isHighRisk ? (
            <>
              <button
                onClick={onRunAgent}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${modeMeta.color} bg-surface-hover border border-line hover:border-line-strong`}
              >
                Run {modeMeta.label}
              </button>
              <button
                onClick={onTextOnly}
                className="px-3 py-1.5 rounded-lg text-xs text-ink-muted bg-surface-hover border border-line hover:text-ink hover:border-line-strong transition-colors"
              >
                Answer text-only
              </button>
            </>
          ) : (
            <button
              onClick={onRunAgent}
              className="px-3 py-1.5 rounded-lg text-xs font-medium text-danger bg-danger/10 border border-danger/30 hover:bg-danger/20 transition-colors"
            >
              Review plan &amp; approve
            </button>
          )}
          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg text-xs text-ink-faint hover:text-ink transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Chat artifact card (inline Code / Preview tabs) ──────────────────────────

function ChatArtifactCard({ artifact, onOpenPreview, previewDocked = false }) {
  const [activeTab, setActiveTab]           = useState('preview');
  const [tailwindEnabled, setTailwindEnabled] = useState(
    artifact.kind === 'tailwind_html' ||
    /cdn\.tailwindcss\.com/i.test(artifact.html)
  );

  const srcDoc  = buildPreviewDoc(artifact, tailwindEnabled);
  const sandbox = 'allow-scripts';
  const previewKey = `${artifact.kind}_${artifact.source}_${tailwindEnabled}_${artifact.html.length}`;

  return (
    <div className="mt-2 min-w-0 max-w-full overflow-hidden rounded-xl border border-line bg-surface-raised">
      {/* Tab bar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-line bg-surface-hover">
        <div className="flex items-center gap-1">
          {['preview', 'code'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-2.5 py-1 text-xs rounded capitalize transition-colors ${
                activeTab === tab
                  ? 'bg-surface-hover text-ink'
                  : 'text-ink-faint hover:text-ink'
              }`}
            >
              {tab === 'preview' ? '⬡ Preview' : '{ } Code'}
            </button>
          ))}
          <span className="ml-2 text-[10px] font-mono text-ink-faint border border-line rounded px-1.5 py-0.5">
            {artifact.kind}
          </span>
        </div>
        {activeTab === 'preview' && !previewDocked && (
          <label className="flex items-center gap-1.5 text-xs text-ink-muted cursor-pointer select-none">
            <input
              type="checkbox"
              checked={tailwindEnabled}
              onChange={e => setTailwindEnabled(e.target.checked)}
              className="accent-accent"
            />
            Tailwind CDN
          </label>
        )}
        {activeTab === 'preview' && (
          <button
            type="button"
            onClick={onOpenPreview}
            aria-pressed={previewDocked}
            className="ml-2 rounded border border-line px-2 py-1 text-xs text-ink-muted hover:border-line-strong hover:text-ink transition-colors"
          >
            {previewDocked ? 'Preview docked' : 'Open right'}
          </button>
        )}
      </div>

      {/* Preview tab */}
      {activeTab === 'preview' && (
        <div className="min-w-0 max-w-full overflow-hidden">
          {artifact.warnings.map((w, i) => (
            <p key={i} className="px-3 pt-2 text-xs text-warn">⚠ {w}</p>
          ))}
          {previewDocked ? (
            <div className="flex min-h-24 items-center justify-center px-4 py-7 text-center text-xs text-ink-faint">
              <span>Preview is open in the right inspector.</span>
            </div>
          ) : (
            <iframe
              key={previewKey}
              srcDoc={srcDoc}
              sandbox={sandbox}
              title="Preview"
              style={{ width: '100%', height: 340, border: 'none', display: 'block', maxWidth: '100%', colorScheme: 'light' }}
            />
          )}
        </div>
      )}

      {/* Code tab */}
      {activeTab === 'code' && (
        <pre className="p-4 text-xs text-ink whitespace-pre-wrap break-words font-mono leading-relaxed overflow-x-auto max-h-80">
          {artifact.html}
          {artifact.css ? `\n\n/* CSS */\n${artifact.css}` : ''}
        </pre>
      )}
    </div>
  );
}

// ── Inline agent run card (chat-integrated agents, Part F) ───────────────────

const AGENT_PHASE_META = {
  planning:             { label: 'Planning',             dot: 'bg-info',   color: 'text-info' },
  running_browser:      { label: 'Running browser',      dot: 'bg-accent', color: 'text-accent' },
  running_command:      { label: 'Running command',      dot: 'bg-accent', color: 'text-accent' },
  waiting_for_approval: { label: 'Waiting for approval', dot: 'bg-warn',   color: 'text-warn' },
  failed:               { label: 'Failed',               dot: 'bg-danger', color: 'text-danger' },
  completed:            { label: 'Completed',            dot: 'bg-ok',     color: 'text-ok' },
};

const AGENT_INTENT_LABELS = {
  browser_agent:    'Browser agent',
  terminal_agent:   'Terminal agent',
  multi_tool_agent: 'Multi-tool agent',
};

function AgentRunCard({
  run, onApprove, onDeny, onCaptchaContinue, onUseApiSearch, onStop,
}) {
  const [expanded, setExpanded] = useState(false);
  if (!run) {
    return (
      <div className="mb-4 px-4 py-3 text-xs text-ink-faint">
        <TheriadLoadingAnimation label="Starting your task…" />
      </div>
    );
  }
  const meta = AGENT_PHASE_META[run.phase] ?? AGENT_PHASE_META.planning;
  const trace = run.trace ?? {};
  const pending = trace.pending_approval;
  const urls = trace.urls_visited ?? [];
  const commands = trace.commands ?? [];
  const actions = trace.actions ?? [];
  const searchResults = trace.search_results ?? [];
  const sources = Array.isArray(trace.sources) && trace.sources.length
    ? trace.sources
    : searchResults;
  const displaySources = sources
    .filter(source => /^https?:\/\//i.test(String(source?.url ?? '')))
    .slice(0, 6);
  const isActive = !['completed', 'failed'].includes(run.phase);
  const finalAnswer = trace.final_answer || null;
  const executionKind = trace.execution_kind ?? (
    searchResults.length > 0 && !trace.current_url ? 'web_search' : 'browser_agent'
  );
  const isDirectWebSearch = ['web_search', 'page_fetch'].includes(executionKind)
    || (searchResults.length > 0 && !trace.current_url);
  const modelsUsed = Array.isArray(trace.models_used) ? trace.models_used : [];
  const routedModel = run.model
    || trace.model
    || [...modelsUsed].reverse().find(model => model?.model)?.model
    || null;
  const routedModelName = routedModel ? shortModelName(routedModel) : 'Theriad';
  const modelCalls = Number(trace.model_calls ?? 0);
  const elapsedSeconds = Number(trace.elapsed_seconds);
  const elapsedText = Number.isFinite(elapsedSeconds)
    ? `${elapsedSeconds.toFixed(1)}s`
    : 'unavailable';
  const costKnown = trace.cost_known !== false && trace.estimated_cost_usd != null;
  const costText = costKnown
    ? `$${Number(trace.estimated_cost_usd).toFixed(6)}`
    : 'unavailable';
  const kindLabel = executionKind === 'web_search'
    ? 'Web search'
    : executionKind === 'page_fetch'
      ? 'Page fetch'
      : AGENT_INTENT_LABELS[run.intent] ?? run.intent;

  return (
    <div
      data-testid="agent-run-card"
      data-current-url={trace.current_url || ''}
      className="theriad-message theriad-message--assistant theriad-agent-run"
    >
      <div className="theriad-message__header theriad-agent-run__header">
        <div className="theriad-message__identity">
          {routedModel ? <ModelLogo model={routedModel} size="avatar" /> : (
            <span className="theriad-message__avatar theriad-message__avatar--assistant" aria-hidden="true">T</span>
          )}
          <span className="theriad-message__author">{routedModelName}</span>
          {routedModel && <span className="theriad-message__route-note">(routed to by Theriad)</span>}
        </div>
        <div className="theriad-agent-run__status" aria-label={`${run.phase_label ?? meta.label} · ${kindLabel}`}>
          <span className={`theriad-agent-run__phase ${meta.color}`}>
            <span className={`theriad-agent-run__phase-dot ${meta.dot} ${isActive ? 'animate-pulse' : ''}`} />
            {run.phase_label ?? meta.label}
          </span>
          <span className="theriad-agent-run__kind">{kindLabel}</span>
          {trace.synthesis_status === 'synthesized' && (
            <span className="theriad-agent-run__badge theriad-agent-run__badge--ok">Synthesized</span>
          )}
          {trace.synthesis_status === 'fallback_digest' && (
            <span className="theriad-agent-run__badge theriad-agent-run__badge--warn">Source digest</span>
          )}
          {trace.observation_backend === 'playwright' && (
            <span className="theriad-agent-run__badge theriad-agent-run__badge--ok">Live</span>
          )}
          <button
            type="button"
            onClick={() => setExpanded(v => !v)}
            className="theriad-agent-run__details-toggle"
          >
            {expanded ? 'Hide details ▲' : 'Show details ▼'}
          </button>
        </div>
      </div>

      <div className="theriad-message__body theriad-agent-run__body">
        {isActive && !trace.captcha_detected && (
          <TheriadLoadingAnimation status={agentSynopsis(run, trace)} />
        )}

        {['completed', 'failed'].includes(run.phase) && finalAnswer && (
          <MarkdownContent content={finalAnswer} />
        )}
        {run.phase === 'completed' && !finalAnswer && (
          <p className="rounded-lg bg-warn/10 border border-warn/30 px-3 py-2 text-warn">
            The task completed, but no final answer was generated.
          </p>
        )}

        {run.error && !finalAnswer && <p className="text-danger">{run.error}</p>}
        {trace.captcha_detected && (
          <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 space-y-2">
            <p className="text-warn font-medium">CAPTCHA encountered</p>
            <p className="text-ink">
              {trace.browser_handoff_ready
                ? 'Theriad opened a temporary browser window. Complete the CAPTCHA there, then continue.'
                : 'Theriad tried automatic protection handling but could not clear this challenge.'}
            </p>
            {trace.page_title && <p className="text-ink-muted">{trace.page_title}</p>}
            {trace.screenshot_base64 && (
              <img
                src={`data:image/png;base64,${trace.screenshot_base64}`}
                alt="CAPTCHA browser state"
                className="w-full max-h-64 object-contain object-top rounded border border-line bg-white"
              />
            )}
            <div className="flex flex-wrap gap-2">
              <button onClick={onCaptchaContinue}
                className="px-2.5 py-1 rounded border border-ok/30 text-ok bg-ok/10">
                {trace.browser_handoff_ready ? 'I finished it — continue' : 'Try again'}
              </button>
              <button onClick={onUseApiSearch}
                className="px-2.5 py-1 rounded border border-accent/30 text-accent bg-accent/10">
                Use API search instead
              </button>
              <button onClick={onStop}
                className="px-2.5 py-1 rounded border border-danger/30 text-danger bg-danger/10">
                Stop task
              </button>
            </div>
          </div>
        )}
        {/* Approval gate */}
        {run.phase === 'waiting_for_approval' && pending && (
          <div className="rounded-lg border border-warn/40 bg-warn/10 p-2.5 space-y-2">
            <p className="text-warn font-medium">Approval required: {pending.proposed_action}</p>
            <p className="text-ink-muted">{pending.reason}</p>
            <div className="flex gap-2">
              <button onClick={onApprove}
                className="px-2.5 py-1 rounded-md text-xs font-medium text-ok bg-ok/10 border border-ok/30 hover:bg-ok/20 transition-colors">
                Approve
              </button>
              <button onClick={onDeny}
                className="px-2.5 py-1 rounded-md text-xs font-medium text-danger bg-danger/10 border border-danger/30 hover:bg-danger/20 transition-colors">
                Deny
              </button>
            </div>
          </div>
        )}

        {/* Expandable trace */}
        {expanded && (
          <div className="theriad-agent-run__details space-y-2 pt-1 border-t border-line">
            <div className="flex items-center gap-4 text-ink-muted flex-wrap">
              <span>{isDirectWebSearch ? 'Queries' : 'Steps'}: <span className="text-ink font-mono">{trace.steps_taken ?? 0}</span></span>
              {trace.source_count != null && trace.source_count > 0 && (
                <span>Sources: <span className="text-ink font-mono">{trace.source_count}</span></span>
              )}
              <span>Elapsed: <span className="text-ink font-mono">{elapsedText}</span></span>
              <span>Cost: <span className="text-ink font-mono">{costText}</span></span>
              <span>Models: <span className="text-ink font-mono">{modelsUsed.length || (modelCalls > 0 ? 'unavailable' : 'none')}</span></span>
              {trace.current_url && (
                <span className="truncate max-w-full">URL: <span className="text-ink font-mono">{trace.current_url}</span></span>
              )}
              {commands.length > 0 && (
                <span>Commands: <span className="text-ink font-mono">{commands.length}</span></span>
              )}
            </div>
            {isDirectWebSearch && displaySources.length > 0 && (
              <div className="theriad-web-sources">
                <p className="text-ink-muted font-medium mb-1">Sources</p>
                <div className="space-y-1.5">
                  {displaySources.map((source, index) => {
                    const url = String(source.url);
                    const title = String(source.title || `Source ${index + 1}`).replace(/\]/g, '');
                    const excerpt = source.evidence_excerpt || source.snippet || '';
                    return (
                      <a
                        key={`${url}-${index}`}
                        href={url}
                        target="_blank"
                        rel="noreferrer"
                        className="theriad-web-source"
                      >
                        <span className="theriad-web-source__index">{index + 1}</span>
                        <span className="min-w-0">
                          <strong className="theriad-web-source__title">{title}</strong>
                          {excerpt && <span className="theriad-web-source__excerpt">{String(excerpt)}</span>}
                        </span>
                      </a>
                    );
                  })}
                </div>
              </div>
            )}
            {modelsUsed.length > 0 && (
              <div>
                <p className="text-ink-muted font-medium mb-1">Models used</p>
                {modelsUsed.map((model, i) => (
                  <p key={`${model.provider ?? 'provider'}-${model.model ?? 'model'}-${i}`} className="font-mono text-ink-muted truncate">
                    {model.provider ? `${model.provider} / ` : ''}{model.model ?? 'unknown'}
                    {model.calls ? ` · ${model.calls} call${model.calls === 1 ? '' : 's'}` : ''}
                  </p>
                ))}
              </div>
            )}
            {modelsUsed.length === 0 && modelCalls > 0 && (
              <p className="text-ink-muted">No successful model response was recorded ({modelCalls} call{modelCalls === 1 ? '' : 's'} attempted).</p>
            )}
            {trace.final_result && (
              <pre className="whitespace-pre-wrap break-words font-mono text-ink-muted bg-surface-sunken border border-line rounded-lg p-2 max-h-40 overflow-y-auto">
                {trace.final_result}
              </pre>
            )}
            {urls.length > 0 && (
              <div>
                <p className="text-ink-muted font-medium mb-1">URLs visited</p>
                {urls.map((u, i) => (
                  <p key={i} className="font-mono text-ink-muted truncate">{u}</p>
                ))}
              </div>
            )}
            {commands.length > 0 && (
              <div>
                <p className="text-ink-muted font-medium mb-1">Commands</p>
                {commands.map((c, i) => (
                  <div key={i} className="font-mono text-ink-muted mb-1">
                    <span className="text-ink">{c.command ?? '(no command matched)'}</span>
                    {' — '}
                    <span className={
                      c.verdict === 'blocked' ? 'text-danger'
                      : c.verdict === 'needs_approval' ? 'text-warn'
                      : 'text-ok'
                    }>{c.verdict}</span>
                    {c.executed === false && <span> (not executed)</span>}
                    {c.reason && <span className="block text-[10px] text-ink-faint">{c.reason}</span>}
                    {c.stdout && (
                      <pre className="whitespace-pre-wrap break-words text-[10px] bg-surface-sunken border border-line rounded p-1.5 mt-1 max-h-32 overflow-y-auto">{c.stdout}</pre>
                    )}
                  </div>
                ))}
              </div>
            )}
            {actions.length > 0 && (
              <div>
                <p className="text-ink-muted font-medium mb-1">Steps</p>
                {actions.map((a, i) => (
                  <p key={i} className="text-ink-faint">
                    <span className="font-mono text-ink-muted">#{a.step} {a.action}</span> — {a.result}
                    {a.summary ? ` · ${a.summary}` : ''}
                  </p>
                ))}
              </div>
            )}
            {run.browser_session_id && (
              <p className="text-ink-faint">
                Browser session <span className="font-mono">{run.browser_session_id.slice(0, 8)}…</span>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function previewForMessage(msg) {
  if (!msg || msg.role === 'user' || !msg.content) return null;
  const artifact = extractPreviewArtifact(msg.content);
  if (!artifact) return null;

  return {
    artifact,
    sourceKey: `artifact:${artifact.source}:${artifact.html.length}`,
  };
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ msg, messageKey, onPreviewOpen, previewDocked = false }) {
  const isUser = msg.role === 'user';
  const previewPayload = previewForMessage(msg);
  const artifact = previewPayload?.artifact ?? null;
  const preview = previewPayload?.preview ?? null;
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [reaction, setReaction] = useState(null);
  const storedRoute = msg.run_model ? {
    model: msg.run_model,
    tier: msg.run_tier,
    estimated_cost_usd: msg.run_cost,
  } : null;
  const route = msg.route_details ?? msg.route ?? msg.metadata?.route ?? storedRoute;
  const routeRun = msg.route_run ?? msg.metadata?.run ?? (msg.run_model ? {
    actual_cost_usd: msg.run_cost,
  } : null);
  const routeContext = msg.route_context ?? msg.metadata?.context ?? null;
  const routeCapability = msg.route_capability ?? msg.metadata?.capability ?? null;
  const routeUsage = msg.route_usage ?? null;
  const hasDetails = Boolean(route || routeCapability);
  const routedModel = !isUser ? msg.run_model : null;
  const author = isUser ? 'You' : routedModel ? shortModelName(routedModel) : 'Theriad';
  const messageTime = formatMessageTime(msg.created_at ?? msg.timestamp);

  async function handleCopy() {
    try {
      const didCopy = await copyText(msg.content);
      if (!didCopy) return;
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // Copy is best-effort and should never make a response unusable.
    }
  }

  return (
    <article className={`theriad-message ${isUser ? 'theriad-message--user' : 'theriad-message--assistant'}`}>
      <div className="theriad-message__header">
        <div className="theriad-message__identity">
          {isUser ? (
            <span className="theriad-message__avatar theriad-message__avatar--user" aria-hidden="true">U</span>
          ) : routedModel ? (
            <ModelLogo model={routedModel} size="avatar" />
          ) : (
            <span className="theriad-message__avatar theriad-message__avatar--assistant" aria-hidden="true">T</span>
          )}
          <span className="theriad-message__author">{author}</span>
          {!isUser && routedModel && (
            <span className="theriad-message__route-note">(routed to by Theriad)</span>
          )}
        </div>
        {messageTime && <time className="theriad-message__time">{messageTime}</time>}
      </div>

      <div className="theriad-message__body break-words overflow-x-auto">
        {isUser ? <p className="whitespace-pre-wrap">{msg.content}</p> : <MarkdownContent content={msg.content} />}
      </div>

      <div className="theriad-message__meta-row">
        {!isUser && msg.run_model && (
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-ink-muted">
            {hasDetails && (
              <button
                type="button"
                onClick={() => setDetailsOpen(value => !value)}
                aria-expanded={detailsOpen}
                className="text-accent hover:text-accent-strong underline underline-offset-2 transition-colors"
              >
                details
              </button>
            )}
            {msg.run_cost != null && (
              <span className="text-ok font-mono">· {friendlyCost(Number(msg.run_cost))}</span>
            )}
            {msg.run_tier && (
              <span className="text-ink-faint">· {msg.run_tier}</span>
            )}
            {webActivityLabel(msg.web_activity) && (
              <span className="text-ink-faint">· {webActivityLabel(msg.web_activity)}</span>
            )}
          </div>
        )}
        {!isUser && msg.content && (
          <div className="theriad-message__actions" aria-label="Response actions">
            <button type="button" onClick={handleCopy} className={copied ? 'is-active' : ''} aria-label={copied ? 'Response copied' : 'Copy response'} title={copied ? 'Copied' : 'Copy response'}>
              <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg>
            </button>
            <button type="button" onClick={() => setReaction(reaction === 'up' ? null : 'up')} className={reaction === 'up' ? 'is-active' : ''} aria-label="Helpful response" aria-pressed={reaction === 'up'} title="Helpful">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 10v11H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3Zm0 11h9.5a2 2 0 0 0 1.9-1.4l2.3-7A2 2 0 0 0 18.8 10H14l.7-4.1A3 3 0 0 0 11.8 2L7 10v11Z" /></svg>
            </button>
            <button type="button" onClick={() => setReaction(reaction === 'down' ? null : 'down')} className={reaction === 'down' ? 'is-active' : ''} aria-label="Unhelpful response" aria-pressed={reaction === 'down'} title="Not helpful">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17 14V3h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3Zm0-11H7.5a2 2 0 0 0-1.9 1.4l-2.3 7A2 2 0 0 0 5.2 14H10l-.7 4.1A3 3 0 0 0 12.2 22L17 14V3Z" /></svg>
            </button>
          </div>
        )}
      </div>

      <div>
        {detailsOpen && hasDetails && (
          <div className="mt-3 rounded-xl border border-accent/25 bg-surface-raised overflow-hidden">
            <RouteDetails
              route={route}
              run={routeRun}
              context={routeContext}
              capability={routeCapability}
              usage={routeUsage}
              preview={null}
            />
          </div>
        )}
        {artifact && (
          <ChatArtifactCard
            artifact={artifact}
            onOpenPreview={() => onPreviewOpen?.({
              id: messageKey,
              ...previewPayload,
            })}
            previewDocked={previewDocked}
          />
        )}
        {preview && !artifact && (
          <>
            {previewDocked ? (
              <p className="mt-2 ml-1 text-xs text-ink-faint">
                Preview is open in the right inspector.
              </p>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => {
                    onPreviewOpen?.({ id: messageKey, ...previewPayload });
                    setPreviewOpen(value => !value);
                  }}
                  className="mt-1 ml-1 text-xs text-accent hover:text-accent-strong underline underline-offset-2"
                >
                  {previewOpen ? 'Hide preview' : 'Open preview inspector'}
                </button>
                {previewOpen && (
                  <WebPreviewPanel srcdoc={preview.srcdoc} onClose={() => setPreviewOpen(false)} />
                )}
              </>
            )}
          </>
        )}
      </div>
    </article>
  );
}

const TASK_QUEUE_STATUS = {
  queued:    { label: 'Up next',       dot: 'bg-info',   color: 'text-info' },
  running:   { label: 'Working',       dot: 'bg-accent', color: 'text-accent' },
  waiting:   { label: 'Needs you',     dot: 'bg-warn',   color: 'text-warn' },
  completed: { label: 'Finished',      dot: 'bg-ok',     color: 'text-ok' },
  failed:    { label: 'Failed',        dot: 'bg-danger', color: 'text-danger' },
  cancelled: { label: 'Cancelled',     dot: 'bg-ink-faint', color: 'text-ink-faint' },
};

function queueTaskPreview(text, maxLength = 92) {
  const normalized = String(text ?? '').replace(/\s+/g, ' ').trim();
  return normalized.length > maxLength
    ? `${normalized.slice(0, maxLength - 1)}…`
    : normalized;
}

function TaskQueuePanel({ items, onRemoveQueued, onClearFinished }) {
  const activeItems = items.filter(item => ['queued', 'running', 'waiting'].includes(item.status));
  const queuedItems = items.filter(item => item.status === 'queued');
  const hasActiveTask = items.some(item => ['running', 'waiting'].includes(item.status));

  // A single request is normal chat state, not a queue. Only expose this
  // panel once another prompt is waiting behind work that is already running.
  if (!hasActiveTask || !queuedItems.length) return null;

  const recentItems = items
    .filter(item => ['completed', 'failed', 'cancelled'].includes(item.status))
    .slice(-3);
  const visibleItems = [...activeItems, ...recentItems];
  const queuedCount = items.filter(item => item.status === 'queued').length;
  const hasFinished = recentItems.length > 0;

  return (
    <section className="theriad-task-queue" data-testid="task-queue" aria-label="Task queue">
      <div className="theriad-task-queue__header">
        <div className="flex items-center gap-2 min-w-0">
          <span className="theriad-task-queue__mark" aria-hidden="true">+</span>
          <div className="min-w-0">
            <h2 className="theriad-task-queue__title">Task queue</h2>
            <p className="theriad-task-queue__hint">
              {queuedCount > 0
                ? `${queuedCount} task${queuedCount === 1 ? '' : 's'} waiting in order`
                : 'New prompts can keep joining while this runs'}
            </p>
          </div>
        </div>
        {hasFinished && (
          <button
            type="button"
            onClick={onClearFinished}
            className="theriad-task-queue__clear"
          >
            Clear finished
          </button>
        )}
      </div>

      <div className="theriad-task-queue__list" aria-live="polite">
        {visibleItems.map(item => {
          const meta = TASK_QUEUE_STATUS[item.status] ?? TASK_QUEUE_STATUS.queued;
          const canRemove = item.status === 'queued';
          return (
            <div
              key={item.id}
              className="theriad-task-queue__item"
              data-status={item.status}
              data-testid="task-queue-item"
            >
              <span className={`theriad-task-queue__dot ${meta.dot} ${item.status === 'running' ? 'animate-pulse' : ''}`} />
              <span className={`theriad-task-queue__status ${meta.color}`}>{meta.label}</span>
              <span className="theriad-task-queue__text" title={item.message}>
                {queueTaskPreview(item.message)}
              </span>
              {canRemove && (
                <button
                  type="button"
                  onClick={() => onRemoveQueued(item.id)}
                  className="theriad-task-queue__remove"
                  aria-label={`Remove queued task: ${queueTaskPreview(item.message, 48)}`}
                  title="Remove from queue"
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── Composer ──────────────────────────────────────────────────────────────────

function Composer({
  prompt, setPrompt,
  memoryMentions, attachedFiles, onDetach, onFileInput, loading,
  onSend, onRouteOnly, routeOnlyAvailable,
  queueItems, onRemoveQueued, onClearFinished,
}) {
  const textRef = useRef(null);
  const overlayRef = useRef(null);
  const mentionRef = useRef(null);
  const fileInputRef = useRef(null);
  const [mentionOpen, setMentionOpen] = useState(false);
  const [mentionQuery, setMentionQuery] = useState('');
  const [mentionIndex, setMentionIndex] = useState(0);
  const [actionsOpen, setActionsOpen] = useState(false);

  const filteredMemoryMentions = memoryMentions.filter(mention =>
    mention.alias.startsWith(mentionQuery) ||
    mention.label.toLowerCase().startsWith(mentionQuery)
  );

  useEffect(() => {
    if (!prompt) setMentionOpen(false);
  }, [prompt]);

  function insertMention(mention) {
    const el = textRef.current;
    if (!el || !mention) return;

    const cursor = el.selectionStart;
    const before = prompt.slice(0, cursor);
    const after = prompt.slice(cursor);
    const atIndex = before.lastIndexOf('@');
    if (atIndex < 0) return;

    const nextPrompt = `${before.slice(0, atIndex)}@${mention.alias} ${after}`;
    const nextCursor = atIndex + mention.alias.length + 2;
    setPrompt(nextPrompt);
    setMentionOpen(false);

    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(nextCursor, nextCursor);
    });
  }

  function handlePromptChange(e) {
    const value = e.target.value;
    const cursor = e.target.selectionStart;
    const beforeCursor = value.slice(0, cursor);
    const mentionMatch = beforeCursor.match(/@([A-Za-z0-9_-]*)$/);

    setPrompt(value);
    if (mentionMatch) {
      setMentionOpen(true);
      setMentionQuery(mentionMatch[1].toLowerCase());
      setMentionIndex(0);
    } else {
      setMentionOpen(false);
    }
  }

  function handleKey(e) {
    if (mentionOpen && filteredMemoryMentions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setMentionIndex(index => Math.min(index + 1, filteredMemoryMentions.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setMentionIndex(index => Math.max(index - 1, 0));
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        insertMention(filteredMemoryMentions[mentionIndex]);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setMentionOpen(false);
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  }

  function handleFileChange(e) {
    const files = [...e.target.files];
    onFileInput(files);
    e.target.value = '';
  }

  function insertCodeBlock() {
    const el = textRef.current;
    if (!el) return;
    const start = el.selectionStart ?? prompt.length;
    const end = el.selectionEnd ?? start;
    const insertion = '```\n\n```';
    const nextPrompt = `${prompt.slice(0, start)}${insertion}${prompt.slice(end)}`;
    setPrompt(nextPrompt);
    setActionsOpen(false);
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(start + 4, start + 4);
    });
  }

  return (
    <div className="theriad-composer border-t border-line px-4 pb-4 pt-3 flex-shrink-0">
      <TaskQueuePanel
        items={queueItems}
        onRemoveQueued={onRemoveQueued}
        onClearFinished={onClearFinished}
      />

      {/* File chips */}
      {attachedFiles.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {attachedFiles.map(f => (
            <span
              key={f.id}
              className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-hover border border-line text-xs text-ink"
            >
              {f.original_name}
              <button
                onClick={() => onDetach(f.id)}
                className="text-ink-faint hover:text-ink ml-0.5 leading-none"
              >✕</button>
            </span>
          ))}
        </div>
      )}

      {/* Input surface: text first, tools below, matching the compact workspace
          composer while keeping the existing upload/mention behavior. */}
      <div className="theriad-composer__surface relative">
        {mentionOpen && filteredMemoryMentions.length > 0 && (
          <div
            ref={mentionRef}
            role="listbox"
            aria-label="Memory uploads"
            className="absolute bottom-full left-0 mb-2 w-64 bg-surface-raised border border-line-strong rounded-xl shadow-xl z-50 overflow-hidden"
          >
            {filteredMemoryMentions.map((mention, index) => (
              <button
                key={mention.id}
                type="button"
                role="option"
                aria-selected={index === mentionIndex}
                onMouseEnter={() => setMentionIndex(index)}
                onMouseDown={e => {
                  e.preventDefault();
                  insertMention(mention);
                }}
                className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                  index === mentionIndex ? 'bg-surface-hover' : 'hover:bg-surface-hover'
                }`}
              >
                <span className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: MEMORY_COLORS[index % MEMORY_COLORS.length] }}
                  />
                  <span style={{ color: MEMORY_COLORS[index % MEMORY_COLORS.length] }}>@{mention.alias}</span>
                  <span className="text-ink-faint text-xs ml-auto">{mention.label}</span>
                </span>
                <span className="block pl-4 text-[11px] text-ink-faint mt-0.5">
                  {mention.count.toLocaleString()} {mention.count === 1 ? 'memory' : 'memories'}
                  {mention.sources?.length ? ` · ${mention.sources.join(', ')}` : ''}
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="relative min-w-0">
          <div
            ref={overlayRef}
            aria-hidden="true"
            className="theriad-composer-overlay absolute inset-0 bg-surface-raised border border-transparent rounded-xl px-4 py-3 text-sm text-ink leading-relaxed whitespace-pre-wrap break-words overflow-hidden pointer-events-none"
            style={{ fontFamily: 'inherit' }}
          >
            {tokenizePrompt(`${prompt} `, memoryMentions).map((token, index) => (
              token.color ? (
                <span key={index} style={{ color: token.color, fontWeight: 500 }}>
                  {token.text}
                </span>
              ) : (
                <span key={index}>{token.text}</span>
              )
            ))}
          </div>
          <textarea
            data-testid="chat-composer"
            ref={textRef}
            value={prompt}
            onChange={handlePromptChange}
            onKeyDown={handleKey}
            onScroll={e => {
              if (!overlayRef.current) return;
              overlayRef.current.scrollTop = e.currentTarget.scrollTop;
              overlayRef.current.scrollLeft = e.currentTarget.scrollLeft;
            }}
            placeholder="Ask anything..."
            aria-label="Message Theriad"
            rows={3}
            className="theriad-composer-input relative block w-full bg-transparent border-0 rounded-t-xl px-4 py-3 text-sm leading-relaxed resize-none focus:outline-none placeholder-ink-faint"
            style={{ color: 'transparent', caretColor: 'var(--color-ink)' }}
          />
        </div>
        <div className="theriad-composer__toolbar">
          <div className="theriad-composer__tools">
            <div className="relative">
              <button
                type="button"
                onClick={() => setActionsOpen(value => !value)}
                aria-expanded={actionsOpen}
                aria-haspopup="menu"
                className="theriad-composer-tool"
                title="More composer actions"
                aria-label="More composer actions"
              >
                <span aria-hidden="true">+</span>
              </button>
              {actionsOpen && (
                <div className="theriad-composer-menu" role="menu">
                  <button type="button" role="menuitem" onClick={() => { fileInputRef.current?.click(); setActionsOpen(false); }}>
                    Attach a file
                  </button>
                  <button type="button" role="menuitem" onClick={insertCodeBlock}>
                    Insert code block
                  </button>
                  {routeOnlyAvailable && (
                    <button type="button" role="menuitem" disabled={!prompt.trim()} onClick={() => { onRouteOnly(); setActionsOpen(false); }}>
                      Preview route only
                    </button>
                  )}
                </div>
              )}
            </div>
            <button type="button" onClick={insertCodeBlock} className="theriad-composer-tool theriad-composer-tool--code" title="Insert code block" aria-label="Insert code block">
              <span aria-hidden="true">&lt;/&gt;</span>
            </button>
            {/* Attach file — works in quick chat and in projects */}
          <label
            title="Attach file"
            htmlFor="theriad-chat-file-input"
            className="theriad-composer-tool cursor-pointer"
          >
            <input id="theriad-chat-file-input" ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange} multiple
              accept={[...ALLOWED_EXTENSIONS].join(',')} />
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" className="w-4 h-4" aria-hidden="true">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
          </label>
          </div>
          {/* Send */}
          <button
            type="button"
            onClick={e => {
              e.preventDefault();
              onSend();
            }}
            disabled={!prompt.trim()}
            className="theriad-send-button flex items-center justify-center w-9 h-9 rounded-lg bg-accent text-on-accent hover:bg-accent-strong disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Queue task (Enter)"
            aria-label="Queue task"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" className="w-4 h-4">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
      </div>

      <div className="theriad-composer-footer flex items-center justify-between gap-3 text-[10px] text-ink-faint">
        <span>{loading ? 'Task running · new prompts join the queue' : 'Enter to send, Shift + Enter for new line'}</span>
        {queueItems.some(item => item.status === 'queued') && (
          <span className="font-mono text-accent">
            {queueItems.filter(item => item.status === 'queued').length} waiting
          </span>
        )}
      </div>

    </div>
  );
}

function IconPlaceholder() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function ChatHistoryPanel({
  conversations,
  activeConvId,
  loading,
  hasMore,
  search,
  onSearchChange,
  onSelect,
  onNewChat,
  onLoadMore,
  open,
  onClose,
  panelRef,
}) {
  const conversationGroups = [];
  const groupsByKey = new Map();
  conversations.forEach(conversation => {
    const projectName = typeof conversation.project_name === 'string'
      ? conversation.project_name.trim()
      : '';
    const groupKey = conversation.project_id
      ? `project:${conversation.project_id}`
      : projectName
        ? `project-name:${projectName.toLowerCase()}`
        : 'unassigned';
    let group = groupsByKey.get(groupKey);
    if (!group) {
      group = {
        key: groupKey,
        title: projectName || (conversation.project_id ? 'Unnamed project' : 'No project'),
        conversations: [],
      };
      groupsByKey.set(groupKey, group);
      conversationGroups.push(group);
    }
    group.conversations.push(conversation);
  });

  return (
    <aside
      ref={panelRef}
      id="theriad-chat-history-panel"
      className={`theriad-chat-history ${open ? 'is-open' : 'is-collapsed'}`}
      aria-label="Chat history"
      data-testid="chat-history"
      data-open={open ? 'true' : 'false'}
    >
      <div className="theriad-chat-history__header">
        <div>
          <h2>Conversations</h2>
        </div>
        <div className="theriad-chat-history__actions">
          <button
            type="button"
            onClick={onNewChat}
            className="theriad-chat-history__new"
            aria-label="Start a new chat"
            title="New chat"
          >
            +
          </button>
          <button
            type="button"
            onClick={onClose}
            className="theriad-chat-history__close"
            aria-label="Close chat history"
            title="Close chat history"
          >
            ×
          </button>
        </div>
      </div>

      <label className="theriad-chat-history__search">
        <span className="sr-only">Search chat history</span>
        <input
          type="search"
          value={search}
          onChange={event => onSearchChange(event.target.value)}
          placeholder="Search saved chats"
          aria-label="Search saved chats"
        />
      </label>

      <div className="theriad-chat-history__list">
        {loading && conversations.length === 0 && (
          <p className="theriad-chat-history__empty">Loading saved chats…</p>
        )}
        {!loading && conversations.length === 0 && (
          <p className="theriad-chat-history__empty">
            {search.trim() ? 'No saved chats match.' : 'Your saved chats will appear here.'}
          </p>
        )}
        {conversationGroups.map(group => (
          <section
            key={group.key}
            className="theriad-chat-history__group"
            aria-label={`${group.title} conversations`}
          >
            <h3 className="theriad-chat-history__group-title">
              {group.title}
            </h3>
            {group.conversations.map(conversation => {
              const conversationTitle = conversation.title || 'Untitled conversation';
              const conversationPreview = conversation.snippet || conversation.first_message || conversationTitle;
              const messageCount = conversation.message_count
                ? `, ${conversation.message_count} messages`
                : '';
              return (
                <button
                  type="button"
                  key={conversation.id}
                  onClick={() => onSelect(conversation)}
                  className={`theriad-chat-history__item ${activeConvId === conversation.id ? 'is-active' : ''}`}
                  aria-current={activeConvId === conversation.id ? 'page' : undefined}
                  aria-label={`${conversationTitle}${messageCount}, ${group.title}`}
                  title={conversationPreview}
                >
                  <span className="theriad-chat-history__item-title">
                    {conversationTitle}
                  </span>
                </button>
              );
            })}
          </section>
        ))}
        {loading && conversations.length > 0 && (
          <p className="theriad-chat-history__empty">Loading more saved chats…</p>
        )}
        {!loading && hasMore && (
          <button
            type="button"
            onClick={onLoadMore}
            className="theriad-chat-history__load-more"
          >
            Load more
          </button>
        )}
      </div>
    </aside>
  );
}

// ── Chat page ─────────────────────────────────────────────────────────────────

export default function ChatPage({
  initialConvId,
  blankChat = false,
  newChatRequest = 0,
  preview = false,
  isGuest = false,
}) {
  // Conversations are surfaced through the consolidated app rail.
  const [activeConvId, setActiveConvId]       = useState(initialConvId ?? null);
  const [activeConvTitle, setActiveConvTitle] = useState('');

  // Messages
  const [messages, setMessages]   = useState([]);
  const [dockedPreview, setDockedPreview] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(!preview);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historySearch, setHistorySearch] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const historyControlRef = useRef(null);
  const historyPanelRef = useRef(null);
  const historyToggleRef = useRef(null);
  const [chatHoverActive, setChatHoverActive] = useState(false);
  // Chat-integrated agent runs: run_id → latest status payload (AgentRunCard).
  const [agentRuns, setAgentRuns] = useState({});
  const msgEndRef                 = useRef(null);

  useEffect(() => {
    if (!historyOpen) return undefined;
    const closeOnOutsidePress = event => {
      if (
        !historyControlRef.current?.contains(event.target) &&
        !historyPanelRef.current?.contains(event.target)
      ) setHistoryOpen(false);
    };
    const closeOnEscape = event => {
      if (event.key !== 'Escape') return;
      setHistoryOpen(false);
      historyToggleRef.current?.focus();
    };
    document.addEventListener('pointerdown', closeOnOutsidePress);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePress);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [historyOpen]);

  // Projects
  const [projects, setProjects]         = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(null);

  // Composer
  const [prompt, setPrompt]         = useState('');
  const policy = 'balanced';
  // Keep the wire fields at their existing defaults while routing stays automatic.
  const speedPreference = 50;
  const qualityPreference = 50;
  const [attachedFiles, setAttachedFiles] = useState([]);  // {id, original_name}
  const [runtimeSettings, setRuntimeSettings] = useState(DEFAULT_RUNTIME_SETTINGS);
  const [memoryMentions, setMemoryMentions] = useState([]);

  // State
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);

  // Queue state is deliberately client-side and FIFO. Each item captures the
  // conversation/project/files that were selected when it was added so later
  // edits in the composer cannot change an already queued request.
  const [queueItems, setQueueItems] = useState([]);
  const queueItemsRef = useRef([]);
  const queueWorkerRef = useRef(false);
  const activeQueueTaskRef = useRef(null);
  const queueSequenceRef = useRef(0);
  const queueScopeRef = useRef(`queue-scope-${Date.now()}`);
  const conversationByQueueScopeRef = useRef(new Map());
  const queueDecisionResolversRef = useRef(new Map());
  const activeConvIdRef = useRef(initialConvId ?? null);
  activeConvIdRef.current = activeConvId;

  const [usageSummary, setUsageSummary]           = useState(null);

  // Capability recommendation (shown inline in chat when agentic mode detected)
  // { capability: CapabilityResult, savedPrompt: string }
  const [pendingCapability, setPendingCapability] = useState(null);
  const [pendingCost, setPendingCost] = useState(null);

  const [routingMenuOpen, setRoutingMenuOpen] = useState(false);
  const [costMenuOpen, setCostMenuOpen] = useState(false);
  const [headerMenuOpen, setHeaderMenuOpen] = useState(false);

  const loadConversation = useCallback(async conv => {
    setHistoryOpen(false);
    activeConvIdRef.current = conv.id;
    setActiveConvId(conv.id);
    setActiveConvTitle(conv.title || '');
    if (conv.project_id !== undefined) setSelectedProjectId(conv.project_id || null);
    setMessages([]);
    try {
      const data = await apiFetch(`/api/conversations/${conv.id}`);
      if (data.project_id !== undefined) setSelectedProjectId(data.project_id || null);
      setActiveConvTitle(data.title || conv.title || '');
      setMessages(data.messages ?? []);
    } catch (e) {
      if (!(isGuest && e.status === 401)) setError(toFriendlyError(e));
    }
  }, [isGuest]);

  // ── Load on mount ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (preview) return undefined;

    apiFetch('/api/projects').then(setProjects).catch(() => {});
    apiFetch('/api/settings/runtime').then(setRuntimeSettings).catch(() => {});
    apiFetch('/api/memory/mentions').then(data => setMemoryMentions(data.mentions ?? [])).catch(() => {});
    refreshUsageSummary();
  }, [preview]);

  const loadConversationHistory = useCallback(async ({
    openLatest = false,
    query = '',
    append = false,
    offset = 0,
  } = {}) => {
    if (preview) return;
    setHistoryLoading(true);
    try {
      const params = new URLSearchParams({
        limit: '100',
        ...(query.trim() ? { q: query.trim() } : {}),
        ...(append ? { offset: String(offset) } : {}),
      });
      const data = await apiFetch(`/api/conversations?${params.toString()}`);
      const rows = data.items ?? [];
      setConversations(previous => append ? [...previous, ...rows] : rows);
      setHistoryHasMore(Boolean(data.has_more));
      // A returning user should see the durable conversation immediately. Keep
      // an explicit Recents selection and an in-progress/new chat authoritative.
      if (openLatest && !initialConvId && !activeConvIdRef.current && rows[0]) {
        await loadConversation(rows[0]);
      }
    } catch (e) {
      setHistoryHasMore(false);
      if (isGuest && e.status === 401) {
        setConversations([]);
      } else {
        setError(toFriendlyError(e));
      }
    } finally {
      setHistoryLoading(false);
    }
  }, [initialConvId, isGuest, loadConversation, preview]);

  useEffect(() => {
    if (preview) return undefined;
    const delay = historySearch.trim() ? 180 : 0;
    const timer = window.setTimeout(() => {
      void loadConversationHistory({
        openLatest: !initialConvId && !historySearch.trim() && !blankChat,
        query: historySearch,
      });
    }, delay);
    return () => window.clearTimeout(timer);
  }, [blankChat, historySearch, initialConvId, loadConversationHistory, preview]);

  function refreshUsageSummary() {
    apiFetch('/api/usage/summary').then(setUsageSummary).catch(() => {});
  }

  // Recents can navigate to Chat with an id before Chat has mounted. Load the
  // requested conversation explicitly instead of relying on a second click.
  useEffect(() => {
    if (initialConvId) void loadConversation({ id: initialConvId, title: '' });
  }, [initialConvId, loadConversation]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messages.length === 0) return;
    msgEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load project files when project changes
  useEffect(() => {
    setAttachedFiles([]);
  }, [selectedProjectId]);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function handleNewChat() {
    setHistoryOpen(false);
    activeConvIdRef.current = null;
    setActiveConvId(null);
    setActiveConvTitle('');
    setMessages([]);
    setHistorySearch('');
    setPrompt('');
    setAttachedFiles([]);
    queueScopeRef.current = `queue-scope-${Date.now()}-${queueSequenceRef.current + 1}`;
    setError(null);
  }

  useEffect(() => {
    if (newChatRequest > 0) handleNewChat();
  }, [newChatRequest]);

  useEffect(() => {
    const onNewChat = () => handleNewChat();
    window.addEventListener('theriad:new-chat', onNewChat);
    return () => window.removeEventListener('theriad:new-chat', onNewChat);
  }, []);

  // ── File handling ────────────────────────────────────────────────────────────
  function handleFileUploaded(file) {
    // Auto-attach newly uploaded files
    setAttachedFiles(prev =>
      prev.find(f => f.id === file.id) ? prev : [...prev, file]
    );
  }

  async function handleFileInput(files) {
    for (const file of files) {
      if (!ALLOWED_EXTENSIONS.has(extOf(file.name))) {
        setError(`Unsupported: ${file.name}`); continue;
      }
      if (isImageFile(file.name) && file.size > MAX_IMAGE_BYTES) {
        setError(
          `Too large: ${file.name} (${(file.size / 1048576).toFixed(1)} MB, max ${MAX_IMAGE_BYTES / 1048576} MB)`
        );
        continue;
      }
      try {
        const fd = new FormData();
        fd.append('file', file);
        const uploadPath = selectedProjectId
          ? `/api/projects/${selectedProjectId}/files`
          : '/api/files';
        const res = await fetch(uploadPath, { method: 'POST', body: fd });
        if (!res.ok) {
          const body = await res.text().catch(() => res.statusText);
          const uploadError = new Error(`API ${res.status} on POST ${uploadPath}: ${body}`);
          uploadError.status = res.status;
          throw uploadError;
        }
        const saved = await res.json();
        handleFileUploaded(saved);
      } catch (e) {
        setError(toFriendlyError(e));
      }
    }
  }

  // ── Send logic ───────────────────────────────────────────────────────────────

  // Execution modes that need the recommendation card before proceeding.
  const AGENTIC_MODES = new Set(['browser_agent', 'local_tool_agent', 'workflow_agent', 'hybrid']);

  function buildRouteDetails(data, capability) {
    if (!data.route) return null;
    const browserIntent = data.chat_intent === 'browser_agent' ||
      capability?.recommended_execution_mode === 'browser_agent';

    return {
      ...data.route,
      intent: data.chat_intent ?? capability?.recommended_execution_mode ?? null,
      policy,
      speed_preference: speedPreference,
      quality_preference: qualityPreference,
      decision_source: data.chat_intent ? 'deterministic_rules' : null,
      browser_confidence: browserIntent && typeof capability?.confidence === 'number'
        ? capability.confidence
        : undefined,
      memory_nodes: data.route.memory_nodes ?? [],
    };
  }

  function syncQueueItems(nextItems) {
    queueItemsRef.current = nextItems;
    setQueueItems(nextItems);
  }

  function updateQueueItem(taskId, patch) {
    syncQueueItems(queueItemsRef.current.map(item => (
      item.id === taskId ? { ...item, ...patch } : item
    )));
  }

  function conversationIdForTask(task) {
    return task.conversationId
      || (task.queueScopeId && conversationByQueueScopeRef.current.get(task.queueScopeId))
      || null;
  }

  function taskIsVisible(task, conversationId = conversationIdForTask(task)) {
    if (!task) return true;
    if (task.conversationId || conversationId) {
      return activeConvIdRef.current === conversationId;
    }
    return task.queueScopeId === queueScopeRef.current;
  }

  function recordTaskConversation(task, conversationId) {
    if (!conversationId) return;
    if (task.queueScopeId) {
      conversationByQueueScopeRef.current.set(task.queueScopeId, conversationId);
    }
    // Do not pull an old queued task into a newer chat the user opened while it
    // was running. The result remains available in the saved conversation.
    if (
      !activeConvIdRef.current &&
      task.queueScopeId === queueScopeRef.current
    ) {
      activeConvIdRef.current = conversationId;
      setActiveConvId(conversationId);
    }
  }

  function taskPayload(task, executeMode, costDecision = null) {
    return {
      message: task.message,
      project_id: task.projectId,
      conversation_id: conversationIdForTask(task),
      file_ids: task.fileIds,
      policy,
      speed_preference: speedPreference,
      quality_preference: qualityPreference,
      mode: executeMode,
      allow_paid_calls: executeMode === 'execute',
      allow_follow_up_questions: task.allowFollowUpQuestions,
      cost_decision: costDecision,
    };
  }

  function queueCancellation() {
    const error = new Error('Task cancelled');
    error.queueCancelled = true;
    return error;
  }

  function waitForQueueDecision(task, type, payload) {
    return new Promise(resolve => {
      queueDecisionResolversRef.current.set(task.id, { type, resolve });
      updateQueueItem(task.id, { status: 'waiting' });
      if (type === 'cost') {
        setPendingCost({ taskId: task.id, preview: payload, message: task.message });
      } else {
        setPendingCapability({
          taskId: task.id,
          capability: payload,
          savedPrompt: task.message,
        });
      }
    });
  }

  function resolveQueueDecision(taskId, type, decision) {
    const pending = queueDecisionResolversRef.current.get(taskId);
    if (!pending || pending.type !== type) return false;
    queueDecisionResolversRef.current.delete(taskId);
    pending.resolve(decision);
    return true;
  }

  function enqueueTask(mode = 'auto') {
    const message = prompt.trim();
    if (!message) return;

    queueSequenceRef.current += 1;
    const task = {
      id: `task-${Date.now()}-${queueSequenceRef.current}`,
      message,
      mode,
      status: 'queued',
      enqueuedAt: Date.now(),
      projectId: selectedProjectId || null,
      conversationId: activeConvIdRef.current,
      queueScopeId: activeConvIdRef.current ? null : queueScopeRef.current,
      fileIds: attachedFiles.map(file => file.id),
      regularChatReturnsAnswer: runtimeSettings.regular_chat_returns_answer,
      allowFollowUpQuestions: runtimeSettings.allow_follow_up_questions,
    };

    syncQueueItems([...queueItemsRef.current, task]);
    setPrompt('');
    setError(null);
    void drainQueue();
  }

  function removeQueuedTask(taskId) {
    const item = queueItemsRef.current.find(task => task.id === taskId);
    if (item?.status !== 'queued') return;
    syncQueueItems(queueItemsRef.current.filter(task => task.id !== taskId));
  }

  function clearFinishedTasks() {
    syncQueueItems(queueItemsRef.current.filter(task => (
      !['completed', 'failed', 'cancelled'].includes(task.status)
    )));
  }

  async function _doSendCore(task, executeMode, skipCapabilityCheck, costDecision = null) {
    if (!task?.message) return;
    updateQueueItem(task.id, { status: 'running', error: null });
    setError(null);
    setPendingCapability(null);
    setLoading(true);

    const conversationId = conversationIdForTask(task);
    const tempUserMsg = { id: `tmp-${task.id}`, role: 'user', content: task.message };
    if (taskIsVisible(task, conversationId)) {
      setMessages(prev => [...prev, tempUserMsg]);
    }

    try {
      const data = await apiFetch('/api/chat/send', {
        method: 'POST',
        body: JSON.stringify(taskPayload(task, executeMode, costDecision)),
      });
      const resolvedConversationId = data.conversation_id ?? conversationId;
      recordTaskConversation(task, resolvedConversationId);
      void loadConversationHistory({ query: historySearch });

      // Browser tasks launch inline immediately. Runtime permission gates still
      // pause before any sensitive action, so no extra confirmation card is
      // needed merely to begin low-risk navigation/research.
      const cap = data.capability;
      const enrichedRoute = buildRouteDetails(data, cap);
      if (
        !skipCapabilityCheck &&
        data.chat_intent === 'browser_agent' &&
        data.status === 'agent_required'
      ) {
        if (taskIsVisible(task, resolvedConversationId)) {
          setMessages(prev => {
            const replaced = prev.filter(m => m.id !== tempUserMsg.id);
            return [...replaced, data.user_message];
          });
        }
        const agentResult = await startAgentRun(task.message, resolvedConversationId, task);
        if (agentResult?.phase === 'failed') {
          throw new Error(agentResult.error || 'The agent task failed.');
        }
        return;
      }

      // Other agentic modes keep the recommendation card for now.
      if (
        !skipCapabilityCheck &&
        cap &&
        AGENTIC_MODES.has(cap.recommended_execution_mode) &&
        cap.requires_confirmation_before_start
      ) {
        // Keep the user message in chat but surface the recommendation card.
        // Do NOT remove tempUserMsg — user should see their own message.
        if (taskIsVisible(task, resolvedConversationId)) {
          setMessages(prev => {
            const replaced = prev.filter(m => m.id !== tempUserMsg.id);
            return [...replaced, data.user_message];
          });
        }
        const decision = await waitForQueueDecision(task, 'capability', cap);
        setPendingCapability(null);
        if (decision === 'cancel') throw queueCancellation();
        if (decision === 'agent') {
          const agentResult = await startAgentRun(task.message, resolvedConversationId, task);
          if (agentResult?.phase === 'failed') {
            throw new Error(agentResult.error || 'The agent task failed.');
          }
          return;
        }
        if (taskIsVisible(task, resolvedConversationId) && data.user_message?.id) {
          setMessages(prev => prev.filter(message => message.id !== data.user_message.id));
        }
        await _doSendCore(task, 'execute', true, costDecision);
        return;
      }

      // ── Normal path ────────────────────────────────────────────────────────
      // Replace temp user message with server-confirmed one
      if (taskIsVisible(task, resolvedConversationId)) {
        setMessages(prev => {
          const replaced = prev.filter(m => m.id !== tempUserMsg.id);
          const assistantMessage = data.assistant_message
            ? {
                ...data.assistant_message,
                ...(data.run ? {
                  run_id: data.run.id,
                  run_model: data.run.model,
                  run_tier: data.run.tier,
                  run_cost: data.run.actual_cost_usd,
                  run_provider: data.run.provider,
                } : enrichedRoute ? {
                  run_model: enrichedRoute.model,
                  run_tier: enrichedRoute.tier,
                  run_provider: enrichedRoute.provider,
                } : {}),
                // Route evidence belongs to the answer that produced it. It is
                // intentionally kept client-side here so it can be disclosed
                // inline without a permanent side column.
                route_details: enrichedRoute,
                route_run: data.run ?? null,
                route_context: data.context ?? null,
                route_capability: data.capability ?? null,
                route_usage: usageSummary,
                web_activity: data.web ?? null,
              }
            : null;
          return [...replaced, data.user_message, ...(assistantMessage ? [assistantMessage] : [])];
        });
      }

      if (data.run) refreshUsageSummary();
    } catch (e) {
      if (!e.queueCancelled) setError(toFriendlyError(e));
      if (taskIsVisible(task, conversationIdForTask(task))) {
        setMessages(prev => prev.filter(m => m.id !== tempUserMsg.id));
      }
      throw e;
    } finally {
      setLoading(false);
    }
  }

  async function executeQueueTask(task) {
    if (task.mode === 'dry_run' || !task.regularChatReturnsAnswer) {
      await _doSendCore(task, 'dry_run', false);
      return;
    }

    setLoading(true);
    try {
      const preview = await apiFetch('/api/chat/cost-preview', {
        method: 'POST',
        body: JSON.stringify(taskPayload(task, 'execute')),
      });
      if (preview.requires_confirmation) {
        const decision = await waitForQueueDecision(task, 'cost', preview);
        setPendingCost(null);
        if (decision === 'cancel') throw queueCancellation();
        await _doSendCore(task, 'execute', false, decision);
      } else {
        await _doSendCore(task, 'execute', false);
      }
    } finally {
      setLoading(false);
    }
  }

  async function drainQueue() {
    if (queueWorkerRef.current) return;
    queueWorkerRef.current = true;
    try {
      while (true) {
        const task = queueItemsRef.current.find(item => item.status === 'queued');
        if (!task) break;
        activeQueueTaskRef.current = task;
        updateQueueItem(task.id, { status: 'running', error: null });
        try {
          await executeQueueTask(task);
          updateQueueItem(task.id, { status: 'completed', completedAt: Date.now() });
        } catch (e) {
          updateQueueItem(task.id, {
            status: e.queueCancelled ? 'cancelled' : 'failed',
            error: e.queueCancelled ? null : toFriendlyError(e),
            completedAt: Date.now(),
          });
        } finally {
          activeQueueTaskRef.current = null;
          setLoading(false);
        }
      }
    } finally {
      queueWorkerRef.current = false;
      if (queueItemsRef.current.some(item => item.status === 'queued')) {
        void drainQueue();
      }
    }
  }

  function handleSend() {
    enqueueTask('auto');
  }

  function handleRouteOnly() {
    enqueueTask('dry_run');
  }

  const handleChatCanvasMove = useCallback(event => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100));

    event.currentTarget.style.setProperty('--theriad-chat-hover-x', `${x}%`);
    event.currentTarget.style.setProperty('--theriad-chat-hover-y', `${y}%`);
  }, []);

  function resolveCostDecision(decision) {
    const pending = pendingCost;
    if (!pending) return;
    setPendingCost(null);
    resolveQueueDecision(pending.taskId, 'cost', decision);
  }

  // ── Chat-integrated agent runs ──────────────────────────────────────────────
  // Inline agent execution: /api/chat/agent classifies + runs, the run is
  // rendered as an AgentRunCard message, and status is polled until terminal.
  // Browser diagnostics remain attached to the inline run card.

  const agentPollRef = useRef({});
  const agentWaitersRef = useRef(new Map());

  useEffect(() => () => {
    // Clear all pollers on unmount.
    Object.values(agentPollRef.current).forEach(clearInterval);
    agentWaitersRef.current.forEach(({ reject }) => {
      reject(new Error('The chat was closed before the task finished.'));
    });
    agentWaitersRef.current.clear();
  }, []);

  function stopAgentPolling(runId) {
    if (agentPollRef.current[runId]) {
      clearInterval(agentPollRef.current[runId]);
      delete agentPollRef.current[runId];
    }
  }

  function ensureAgentPolling(runId) {
    if (!runId || agentPollRef.current[runId]) return;
    agentPollRef.current[runId] = setInterval(
      () => refreshAgentRun(runId), 1500,
    );
  }

  async function refreshAgentRun(runId) {
    try {
      const data = await apiFetch(`/api/chat/agent/status?run_id=${encodeURIComponent(runId)}`);
      setAgentRuns(prev => ({ ...prev, [runId]: data }));
      if (['completed', 'failed'].includes(data.phase)) {
        stopAgentPolling(runId);
        const waiter = agentWaitersRef.current.get(runId);
        if (waiter) {
          agentWaitersRef.current.delete(runId);
          waiter.resolve(data);
        }
      }
      return data;
    } catch (error) {
      stopAgentPolling(runId);
      const waiter = agentWaitersRef.current.get(runId);
      if (waiter) {
        agentWaitersRef.current.delete(runId);
        waiter.reject(error);
      }
      return null;
    }
  }

  async function waitForAgentRun(runId, initialData) {
    if (['completed', 'failed'].includes(initialData?.phase)) return initialData;
    return new Promise((resolve, reject) => {
      agentWaitersRef.current.set(runId, { resolve, reject });
      ensureAgentPolling(runId);
    });
  }

  async function startAgentRun(messageText, conversationId, task) {
    try {
      const data = await apiFetch('/api/chat/agent', {
        method: 'POST',
        body: JSON.stringify({
          message: messageText,
          conversation_id: conversationId,
          project_id: task?.projectId ?? selectedProjectId ?? null,
          // D-050: opts in to LLM synthesis of the agent's final answer
          // (same convention as /api/chat/send).
          allow_paid_calls: true,
          stream_progress: true,
        }),
      });
      if (!data.agent_run_id) {
        // Classifier says normal chat — fall back to the standard send path.
        if (task) {
          if (taskIsVisible(task, conversationId)) {
            setMessages(prev => prev.filter(message => message.id !== `tmp-${task.id}`));
          }
          await _doSendCore(task, 'execute', true);
        }
        return data;
      }
      setAgentRuns(prev => ({ ...prev, [data.agent_run_id]: data }));
      if (taskIsVisible(task, conversationId)) {
        setMessages(prev => [
          ...prev,
          { id: `agent-${data.agent_run_id}`, role: 'agent_run', agent_run_id: data.agent_run_id },
        ]);
      }
      return await waitForAgentRun(data.agent_run_id, data);
    } catch (e) {
      setError(toFriendlyError(e));
      throw e;
    }
  }

  async function resolveAgentApproval(run, approved) {
    if (!run?.browser_session_id) return;
    try {
      await apiFetch(approved ? '/api/agent/browser/approve' : '/api/agent/browser/deny', {
        method: 'POST',
        body: JSON.stringify({ session_id: run.browser_session_id }),
      });
      const updated = await refreshAgentRun(run.agent_run_id);
      if (updated && !['completed', 'failed'].includes(updated.phase)) {
        ensureAgentPolling(run.agent_run_id);
      }
    } catch (e) {
      setError(toFriendlyError(e));
    }
  }

  async function resolveCaptcha(run, mode) {
    if (!run?.browser_session_id) return;
    const endpoint = mode === 'continue'
      ? '/api/agent/browser/continue_captcha'
      : mode === 'search'
      ? '/api/agent/browser/use_api_search'
      : '/api/agent/browser/stop';
    try {
      await apiFetch(endpoint, {
        method: 'POST',
        body: JSON.stringify({ session_id: run.browser_session_id }),
      });
      const updated = await refreshAgentRun(run.agent_run_id);
      if (updated && !['completed', 'failed'].includes(updated.phase)) {
        ensureAgentPolling(run.agent_run_id);
      }
    } catch (e) {
      setError(toFriendlyError(e));
    }
  }

  // ── Capability card handlers ──────────────────────────────────────────────────

  function handleCapabilityRunAgent() {
    const pending = pendingCapability;
    if (!pending) return;
    setPendingCapability(null);
    resolveQueueDecision(pending.taskId, 'capability', 'agent');
  }

  function handleCapabilityTextOnly() {
    const pending = pendingCapability;
    if (!pending) return;
    setPendingCapability(null);
    resolveQueueDecision(pending.taskId, 'capability', 'text');
  }

  function handleCapabilityCancel() {
    const pending = pendingCapability;
    if (!pending) return;
    setPendingCapability(null);
    resolveQueueDecision(pending.taskId, 'capability', 'cancel');
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  const title = activeConvTitle || 'Chat';
  const selectedProject = projects.find(project => project.id === selectedProjectId);
  const latestModel = [...messages].reverse().find(message => message.role === 'assistant' && message.run_model)?.run_model;
  const headerCost = usageSummary?.total_cost_usd ?? 0;
  useEffect(() => {
    let nextPreview = null;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const candidate = previewForMessage(messages[index]);
      if (candidate) {
        nextPreview = {
          id: messages[index].id ?? `message-${index}`,
          ...candidate,
        };
        break;
      }
    }

    if (!nextPreview) {
      setDockedPreview(null);
      return;
    }
    setDockedPreview(current => {
      if (current?.id === nextPreview.id && current.sourceKey === nextPreview.sourceKey) {
        return current;
      }
      return nextPreview;
    });
  }, [messages]);

  return (
    // h-full fills the parent <main> which is h-full on the outer shell.
    // overflow-hidden keeps the inner flex children from leaking.
    <div className="theriad-chat-frame flex h-full overflow-hidden">
      {historyOpen && (
        <button
          type="button"
          className="theriad-chat-history-backdrop"
          aria-label="Close chat history"
          tabIndex={-1}
          onClick={() => setHistoryOpen(false)}
        />
      )}
      <ChatHistoryPanel
        panelRef={historyPanelRef}
        conversations={conversations}
        activeConvId={activeConvId}
        loading={historyLoading}
        hasMore={historyHasMore}
        search={historySearch}
        onSearchChange={setHistorySearch}
        onSelect={loadConversation}
        onNewChat={handleNewChat}
        open={historyOpen}
        onClose={() => {
          setHistoryOpen(false);
          historyToggleRef.current?.focus();
        }}
        onLoadMore={() => loadConversationHistory({
          query: historySearch,
          append: true,
          offset: conversations.length,
        })}
      />
      {/* ── Main panel ── */}
      <Dropzone
        projectId={selectedProjectId}
        onUploaded={handleFileUploaded}
      >
        {/* min-w-0 prevents this flex child from overflowing its container */}
        <div className="h-full min-h-0 min-w-0 flex flex-col">
          {/* Header strip */}
          <header className="theriad-chat-header flex items-center justify-between gap-4 flex-shrink-0">
            <div className="theriad-chat-heading min-w-0">
              <div className="theriad-history-control" ref={historyControlRef}>
                <button
                  ref={historyToggleRef}
                  type="button"
                  className="theriad-history-toggle"
                  data-testid="history-toggle"
                  aria-expanded={historyOpen}
                  aria-controls="theriad-chat-history-panel"
                  aria-label={historyOpen ? 'Hide chat history' : 'Show chat history'}
                  title={historyOpen ? 'Hide chat history' : 'Show chat history'}
                  onClick={() => {
                    setHistoryOpen(value => !value);
                    setRoutingMenuOpen(false);
                    setCostMenuOpen(false);
                    setHeaderMenuOpen(false);
                  }}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v14.5A2.5 2.5 0 0 1 17.5 20H4z" />
                    <path d="M4 5.5v13A2.5 2.5 0 0 0 6.5 21H20" />
                    <path d="M8 7h8M8 10.5h6" />
                  </svg>
                  <span>History</span>
                </button>
              </div>
              <div className="theriad-chat-heading__copy min-w-0">
                <h1 className="theriad-chat-title truncate">{title}</h1>
                {selectedProject && <p className="theriad-chat-project truncate">{selectedProject.name}</p>}
              </div>
            </div>
            <div className="theriad-chat-header__controls">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => { setRoutingMenuOpen(value => !value); setCostMenuOpen(false); setHeaderMenuOpen(false); }}
                  className="theriad-model-picker"
                  aria-expanded={routingMenuOpen}
                  aria-haspopup="menu"
                >
                  <span>Auto</span>
                  <span className="theriad-model-picker__separator">·</span>
                  <span className="truncate">{shortModelName(latestModel)}</span>
                  <svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="6 9 12 15 18 9" /></svg>
                </button>
                {routingMenuOpen && (
                  <div className="theriad-header-menu theriad-header-menu--model" role="menu">
                    <p className="theriad-header-menu__eyebrow">Automatic routing</p>
                    <p className="theriad-header-menu__copy">Theriad chooses the best fit for quality, latency, and cost.</p>
                    {latestModel && <p className="theriad-header-menu__current">Latest response · {shortModelName(latestModel)}</p>}
                  </div>
                )}
              </div>

              <div className="relative">
                <button
                  type="button"
                  onClick={() => { setCostMenuOpen(value => !value); setRoutingMenuOpen(false); setHeaderMenuOpen(false); }}
                  className="theriad-cost-pill"
                  aria-expanded={costMenuOpen}
                  aria-label={`Usage total ${formatHeaderCost(headerCost)}`}
                >
                  <span>~{formatHeaderCost(headerCost)}</span>
                  <span className="theriad-info-mark" aria-hidden="true">i</span>
                </button>
                {costMenuOpen && (
                  <div className="theriad-header-menu theriad-header-menu--cost" role="status">
                    <p className="theriad-header-menu__eyebrow">Workspace usage</p>
                    <p className="theriad-header-menu__copy">Recorded routed spend</p>
                    <p className="theriad-header-menu__value">{formatHeaderCost(headerCost)}</p>
                  </div>
                )}
              </div>

              <div className="relative">
                <button
                  type="button"
                  onClick={() => { setHeaderMenuOpen(value => !value); setRoutingMenuOpen(false); setCostMenuOpen(false); }}
                  className="theriad-overflow-button"
                  aria-expanded={headerMenuOpen}
                  aria-haspopup="menu"
                  aria-label="Chat options"
                >
                  <span aria-hidden="true">⋮</span>
                </button>
                {headerMenuOpen && (
                  <div className="theriad-header-menu theriad-header-menu--overflow" role="menu">
                    <button type="button" role="menuitem" onClick={() => { handleNewChat(); setHeaderMenuOpen(false); }}>
                      New chat
                    </button>
                    <label className="theriad-header-menu__field">
                      <span>Project</span>
                      <select value={selectedProjectId ?? ''} onChange={event => setSelectedProjectId(event.target.value || null)}>
                        <option value="">None (quick chat)</option>
                        {projects.map(project => <option key={project.id} value={project.id}>{project.name}</option>)}
                      </select>
                    </label>
                  </div>
                )}
              </div>
            </div>
          </header>

          {/* Messages */}
          <div
            className={`theriad-chat-messages flex-1 overflow-y-auto px-4 py-4 ${chatHoverActive ? 'is-hovering' : ''}`}
            onMouseEnter={() => setChatHoverActive(true)}
            onMouseMove={handleChatCanvasMove}
            onMouseLeave={() => setChatHoverActive(false)}
          >
            {messages.length === 0 && (
              <div className="theriad-empty-state flex flex-col items-center justify-center h-full text-center px-6">
                <div className="theriad-empty-icon" aria-hidden="true"><IconPlaceholder /></div>
                <p className="theriad-empty-title">Start a conversation</p>
                <p className="theriad-empty-copy">Ask a question or add some context to get started.</p>
                <div className="theriad-suggestion-row">
                  {[
                    'Explain how Theriad saves money',
                    'Summarize a long document',
                    'Find today’s weather',
                  ].map(example => (
                    <button key={example} type="button" onClick={() => setPrompt(example)} className="theriad-suggestion">
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg, i) => {
              const messageKey = msg.id ?? `message-${i}`;
              if (msg.role === 'agent_run') {
                return (
                  <AgentRunCard
                    key={messageKey}
                    run={agentRuns[msg.agent_run_id]}
                    onApprove={() => resolveAgentApproval(agentRuns[msg.agent_run_id], true)}
                    onDeny={() => resolveAgentApproval(agentRuns[msg.agent_run_id], false)}
                    onCaptchaContinue={() => resolveCaptcha(agentRuns[msg.agent_run_id], 'continue')}
                    onUseApiSearch={() => resolveCaptcha(agentRuns[msg.agent_run_id], 'search')}
                    onStop={() => resolveCaptcha(agentRuns[msg.agent_run_id], 'stop')}
                  />
                );
              }

              const messagePreview = previewForMessage(msg);
              const previewDocked = Boolean(
                messagePreview &&
                dockedPreview?.id === messageKey &&
                dockedPreview.sourceKey === messagePreview.sourceKey
              );
              return (
                <MessageBubble
                  key={messageKey}
                  msg={msg}
                  messageKey={messageKey}
                  onPreviewOpen={setDockedPreview}
                  previewDocked={previewDocked}
                />
              );
            })}
            {/* Inline capability recommendation card */}
            {pendingCapability && (
              <CapabilityRecommendationCard
                capability={pendingCapability.capability}
                onRunAgent={handleCapabilityRunAgent}
                onTextOnly={handleCapabilityTextOnly}
                onCancel={handleCapabilityCancel}
              />
            )}
            {loading && (
              <div className="flex justify-start mb-4">
                <div className="bg-transparent rounded-2xl px-4 py-2.5 flex items-center gap-2.5">
                  <TheriadLoadingAnimation label="Working on your answer…" />
                </div>
              </div>
            )}
            <div ref={msgEndRef} />
          </div>

          {/* Error */}
          {error && (
            <div className="theriad-chat-error mx-4 mb-2 text-xs text-warn bg-warn/10 border border-warn/20 rounded-lg px-3 py-2 flex items-center justify-between">
              {error}
              <button onClick={() => setError(null)} className="ml-2 text-warn hover:text-warn">✕</button>
            </div>
          )}

          {/* Composer */}
          <Composer
            prompt={prompt}
            setPrompt={setPrompt}
            memoryMentions={memoryMentions}
            attachedFiles={attachedFiles}
            onDetach={id => setAttachedFiles(prev => prev.filter(f => f.id !== id))}
            onFileInput={handleFileInput}
            loading={loading}
            onSend={handleSend}
            onRouteOnly={handleRouteOnly}
            routeOnlyAvailable={runtimeSettings.route_only_available}
            queueItems={queueItems}
            onRemoveQueued={removeQueuedTask}
            onClearFinished={clearFinishedTasks}
          />
        </div>
      </Dropzone>

      {dockedPreview && (
        <PreviewInspector
          key={`${dockedPreview.id}:${dockedPreview.sourceKey}`}
          preview={dockedPreview}
          onClose={() => setDockedPreview(null)}
        />
      )}

      {pendingCost && (
        <CostConfirmationDialog
          preview={pendingCost.preview}
          onPaid={() => resolveCostDecision('paid')}
          onFree={() => resolveCostDecision('free')}
          onCancel={() => resolveCostDecision('cancel')}
          allowQueueing
        />
      )}

    </div>
  );
}
