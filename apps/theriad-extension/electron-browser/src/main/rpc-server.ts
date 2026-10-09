/**
 * Local HTTP RPC server — lets Python (theriad/browser/electron_runtime.py) drive
 * the Electron BrowserView without going through Electron IPC.
 *
 * Binds on 127.0.0.1:47200 (override via THERIAD_ELECTRON_RPC_PORT).
 * All payloads are JSON. Only loopback connections are accepted.
 *
 * Endpoints:
 *   GET  /health            — liveness probe
 *   POST /navigate          — { url } → observe result
 *   POST /observe           — { screenshot? } → DOM + optional screenshot
 *   POST /action/click      — { element_id } or { x, y } → observe result
 *   POST /action/fill       — { element_id, text } → observe result
 *   POST /action/scroll     — { amount, direction? } → observe result
 *   POST /action/key        — { key } → observe result
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { randomBytes, timingSafeEqual } from 'crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { dirname, join } from 'path';
import { BrowserManager } from './browser-manager';

// ── Injected JS snippets ──────────────────────────────────────────────────────

/** Extracts interactive elements and visible text from the current page. */
const OBSERVE_SCRIPT = `
(function(limit) {
  const visible = (el) => {
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return s.visibility !== 'hidden' && s.display !== 'none' &&
      r.width > 0 && r.height > 0;
  };
  const labelFor = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria;
    if (el.id) {
      const label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (label) return (label.innerText || label.textContent || '').trim();
    }
    return (el.closest('label')?.innerText || '').trim();
  };
  const candidates = [...document.querySelectorAll(
    'a[href], button, [role="button"], input, textarea, select'
  )].filter(visible).slice(0, limit);
  const clickables = [];
  const inputs = [];
  candidates.forEach((el, index) => {
    let id = el.getAttribute('data-theriad-id');
    if (!id) {
      id = 'ae-' + (index + 1);
      el.setAttribute('data-theriad-id', id);
    }
    const tag = el.tagName.toLowerCase();
    const type = (el.type || el.getAttribute('type') || '').toLowerCase();
    const label = labelFor(el);
    const text = (el.innerText || el.textContent || el.value || '').trim().slice(0, 160);
    const common = {
      id, tag, type, text, label,
      in_form: Boolean(el.form),
      aria_label: el.getAttribute('aria-label') || '',
      placeholder: el.getAttribute('placeholder') || '',
    };
    if (['input', 'textarea', 'select'].includes(tag) && type !== 'hidden') {
      inputs.push(common);
    }
    if (tag === 'a' || tag === 'button' || el.getAttribute('role') === 'button' ||
        ['button', 'submit', 'reset'].includes(type)) {
      clickables.push({ ...common, href: el.href || '', download: el.hasAttribute('download') });
    }
  });
  const roles = [...document.querySelectorAll(
    'main, nav, header, footer, form, [role], h1, h2, h3'
  )].filter(visible).slice(0, 30).map(el => ({
    role: el.getAttribute('role') || el.tagName.toLowerCase(),
    name: (el.getAttribute('aria-label') || el.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 180),
  }));
  return {
    visible_text: (document.body?.innerText || '').trim().slice(0, 4000),
    clickable_elements: clickables,
    input_fields: inputs,
    accessibility_tree_summary: roles,
  };
})(60)
`;

/** Click an element by its injected data-theriad-id attribute. */
const clickScript = (elementId: string): string => `
(function() {
  const target = ${JSON.stringify(elementId)};
  const el = [...document.querySelectorAll('[data-theriad-id]')]
    .find(node => node.getAttribute('data-theriad-id') === target);
  if (!el) return { success: false, error: 'Element not found: ${elementId}' };
  el.focus();
  el.click();
  return { success: true };
})()
`;

/** Fill a form element by its injected data-theriad-id attribute. */
const fillScript = (elementId: string, text: string): string => `
(function() {
  const target = ${JSON.stringify(elementId)};
  const el = [...document.querySelectorAll('[data-theriad-id]')]
    .find(node => node.getAttribute('data-theriad-id') === target);
  if (!el) return { success: false, error: 'Element not found: ${elementId}' };
  el.focus();
  if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
    el.value = ${JSON.stringify(text)};
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } else if (el.isContentEditable) {
    el.textContent = ${JSON.stringify(text)};
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
  return { success: true };
})()
`;

// ── HTTP helpers ──────────────────────────────────────────────────────────────

async function readBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > 1_000_000) {
        reject(new Error('Request body exceeds 1 MB limit'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {});
      } catch {
        reject(new Error('Invalid JSON in request body'));
      }
    });
    req.on('error', reject);
  });
}

function send(res: ServerResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(data),
    'X-Theriad-Backend': 'electron',
  });
  res.end(data);
}

function rpcToken(): string {
  const configured = process.env.THERIAD_ELECTRON_RPC_TOKEN?.trim();
  if (configured) return configured;
  const tokenPath = join(homedir(), '.theriad', 'electron-rpc-token');
  if (existsSync(tokenPath)) return readFileSync(tokenPath, 'utf8').trim();
  mkdirSync(dirname(tokenPath), { recursive: true, mode: 0o700 });
  const generated = randomBytes(32).toString('base64url');
  writeFileSync(tokenPath, generated, { mode: 0o600 });
  return generated;
}

function authorized(req: IncomingMessage, expected: string): boolean {
  const value = String(req.headers.authorization || '');
  const supplied = value.startsWith('Bearer ') ? value.slice(7) : '';
  const left = Buffer.from(supplied);
  const right = Buffer.from(expected);
  return left.length === right.length && left.length > 0 && timingSafeEqual(left, right);
}

// ── RPC server ────────────────────────────────────────────────────────────────

export function startRpcServer(browserManager: BrowserManager): void {
  const port = parseInt(process.env.THERIAD_ELECTRON_RPC_PORT || '47200', 10);
  const token = rpcToken();

  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    // Only accept loopback connections.
    const remote = req.socket.remoteAddress;
    if (remote !== '127.0.0.1' && remote !== '::1' && remote !== '::ffff:127.0.0.1') {
      send(res, 403, { ok: false, error: 'Forbidden: only localhost connections allowed' });
      return;
    }
    if (!authorized(req, token)) {
      send(res, 401, { ok: false, error: 'Unauthorized local RPC request' });
      return;
    }

    const path = (req.url || '/').split('?')[0];
    const method = req.method || 'GET';

    try {
      // ── GET /health ─────────────────────────────────────────────────────────
      if (method === 'GET' && path === '/health') {
        send(res, 200, { ok: true, version: '1.0', backend: 'electron' });
        return;
      }

      // All other routes are POST + JSON body.
      if (method !== 'POST') {
        send(res, 405, { ok: false, error: `Method ${method} not allowed` });
        return;
      }

      const body = await readBody(req);

      // ── POST /navigate ──────────────────────────────────────────────────────
      if (path === '/navigate') {
        const url = body.url as string | undefined;
        if (!url) { send(res, 400, { ok: false, error: 'url required' }); return; }
        await browserManager.navigateToUrl(url);
        send(res, 200, {
          ok: true,
          url: browserManager.getCurrentUrl(),
          title: browserManager.getPageTitle(),
        });
        return;
      }

      // ── POST /observe ───────────────────────────────────────────────────────
      if (path === '/observe') {
        const wc = (browserManager as any).browserView?.webContents;
        if (!wc) { send(res, 503, { ok: false, error: 'BrowserView not initialized' }); return; }

        const dom = await wc.executeJavaScript(OBSERVE_SCRIPT);
        let screenshotB64: string | null = null;
        if (body.screenshot) {
          const dataUrl: string = await browserManager.captureScreenshot();
          // captureScreenshot() returns a data: URL; strip the header.
          screenshotB64 = dataUrl.replace(/^data:image\/\w+;base64,/, '');
        }
        send(res, 200, {
          ok: true,
          backend: 'electron',
          url: browserManager.getCurrentUrl(),
          title: browserManager.getPageTitle(),
          visible_text: dom.visible_text,
          clickable_elements: dom.clickable_elements,
          input_fields: dom.input_fields,
          accessibility_tree_summary: dom.accessibility_tree_summary,
          screenshot_base64: screenshotB64,
        });
        return;
      }

      // ── POST /action/click ──────────────────────────────────────────────────
      if (path === '/action/click') {
        const wc = (browserManager as any).browserView?.webContents;
        if (!wc) { send(res, 503, { ok: false, error: 'BrowserView not initialized' }); return; }

        if (body.element_id) {
          const result = await wc.executeJavaScript(clickScript(body.element_id as string));
          if (!result.success) { send(res, 400, { ok: false, error: result.error }); return; }
        } else if (body.x !== undefined && body.y !== undefined) {
          await browserManager.clickAt(body.x as number, body.y as number);
        } else {
          send(res, 400, { ok: false, error: 'element_id or x,y required' }); return;
        }

        // Short settle wait then re-observe.
        await new Promise(r => setTimeout(r, 200));
        const dom = await wc.executeJavaScript(OBSERVE_SCRIPT);
        send(res, 200, {
          ok: true,
          url: browserManager.getCurrentUrl(),
          title: browserManager.getPageTitle(),
          ...dom,
        });
        return;
      }

      // ── POST /action/fill ───────────────────────────────────────────────────
      if (path === '/action/fill') {
        const elementId = body.element_id as string | undefined;
        const text = body.text as string | undefined;
        if (!elementId || text === undefined) {
          send(res, 400, { ok: false, error: 'element_id and text required' }); return;
        }
        const wc = (browserManager as any).browserView?.webContents;
        if (!wc) { send(res, 503, { ok: false, error: 'BrowserView not initialized' }); return; }

        const result = await wc.executeJavaScript(fillScript(elementId, text));
        if (!result.success) { send(res, 400, { ok: false, error: result.error }); return; }

        const dom = await wc.executeJavaScript(OBSERVE_SCRIPT);
        send(res, 200, {
          ok: true,
          url: browserManager.getCurrentUrl(),
          title: browserManager.getPageTitle(),
          ...dom,
        });
        return;
      }

      // ── POST /action/scroll ─────────────────────────────────────────────────
      if (path === '/action/scroll') {
        const rawAmount = body.amount as number | undefined ?? 300;
        const direction = (body.direction as string | undefined) === 'up' ? 'up' : 'down';
        await browserManager.scrollPage(direction, Math.abs(rawAmount));

        const wc = (browserManager as any).browserView?.webContents;
        if (!wc) { send(res, 503, { ok: false, error: 'BrowserView not initialized' }); return; }
        const dom = await wc.executeJavaScript(OBSERVE_SCRIPT);
        send(res, 200, {
          ok: true,
          url: browserManager.getCurrentUrl(),
          title: browserManager.getPageTitle(),
          ...dom,
        });
        return;
      }

      // ── POST /action/key ────────────────────────────────────────────────────
      if (path === '/action/key') {
        const key = (body.key as string | undefined) || 'Return';
        const wc = (browserManager as any).browserView?.webContents;
        if (!wc) { send(res, 503, { ok: false, error: 'BrowserView not initialized' }); return; }

        // Electron's sendInputEvent for keyboard events.
        wc.sendInputEvent({ type: 'keyDown', keyCode: key });
        wc.sendInputEvent({ type: 'char', keyCode: key });
        wc.sendInputEvent({ type: 'keyUp', keyCode: key });

        // Wait for any form submission or navigation to settle.
        await new Promise(r => setTimeout(r, 400));
        const dom = await wc.executeJavaScript(OBSERVE_SCRIPT);
        send(res, 200, {
          ok: true,
          url: browserManager.getCurrentUrl(),
          title: browserManager.getPageTitle(),
          ...dom,
        });
        return;
      }

      send(res, 404, { ok: false, error: `Unknown endpoint: ${method} ${path}` });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[RpcServer] Error handling', method, path, msg);
      send(res, 500, { ok: false, error: msg });
    }
  });

  server.listen(port, '127.0.0.1', () => {
    console.log(`[RpcServer] Theriad browser RPC server → http://127.0.0.1:${port}`);
  });

  server.on('error', (err: Error) => {
    console.error('[RpcServer] Failed to start:', err.message);
  });
}
