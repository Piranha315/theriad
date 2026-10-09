import type { TheriadMode, RuntimeProfile } from './types';

export type TheriadChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
};

export type TheriadBrowserContext = {
  url?: string;
  title?: string;
  pageText?: string;
  selectionText?: string;
  domSummary?: unknown;
  screenshotAvailable?: boolean;
  browserMemory?: unknown;
};

export type TheriadRequest = {
  messages: TheriadChatMessage[];
  mode: TheriadMode;
  browserContext?: TheriadBrowserContext;
  runtimeProfile?: RuntimeProfile;
  requestedCapabilities?: string[];
  client: {
    name: 'theriad-extension';
    version?: string;
  };
};

export type TheriadResponse = {
  content: string;
  model?: string;
  provider?: string;
  routeTrace?: unknown;
  cost?: {
    inputTokens?: number;
    outputTokens?: number;
    estimatedUsd?: number;
  };
  warnings?: string[];
  toolCalls?: unknown[];
};

export type TheriadConnectionStatus =
  | 'Connected'
  | 'Backend unavailable'
  | 'Endpoint missing'
  | 'CORS/config error'
  | 'API error';

/** Shared Theriad browser-session vocabulary. Extension-local Gemini runs
 * must not be represented as one of these sessions. */
export const THERIAD_BROWSER_SESSION_STATUSES = [
  'idle', 'starting', 'running', 'waiting_for_approval',
  'completed', 'failed', 'stopped', 'budget_exceeded',
] as const;
export type TheriadBrowserSessionStatus = typeof THERIAD_BROWSER_SESSION_STATUSES[number];

export type TheriadBrowserSession = {
  session_id: string;
  status: TheriadBrowserSessionStatus;
  current_url?: string | null;
  current_page_summary?: string;
  pending_approval?: unknown;
  final_result?: string | null;
};

/** Canonical browser-session adapter. Theriad Browser can use this when it needs an
 * Theriad-managed run; it intentionally does not proxy Gemini calls. */
export async function startTheriadBrowserSession(
  apiUrl: string,
  goal: string,
  startUrl?: string
): Promise<TheriadBrowserSession> {
  const response = await fetch(`${normalizeBaseUrl(apiUrl)}/api/browser/session/start`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ goal, start_url: startUrl || null }),
  });
  if (!response.ok) throw new Error(`Theriad browser session failed (${response.status})`);
  return response.json();
}

export async function getTheriadBrowserSession(apiUrl: string, sessionId: string): Promise<TheriadBrowserSession> {
  const response = await fetch(`${normalizeBaseUrl(apiUrl)}/api/browser/session/${encodeURIComponent(sessionId)}`);
  if (!response.ok) throw new Error(`Theriad browser session unavailable (${response.status})`);
  return response.json();
}

const ENDPOINTS = ['/api/chat/send', '/api/execute', '/api/route'] as const;

export async function testTheriadConnection(apiUrl: string): Promise<{ status: TheriadConnectionStatus; detail?: string }> {
  const baseUrl = normalizeBaseUrl(apiUrl);
  try {
    const health = await fetchWithTimeout(`${baseUrl}/api/health`, { method: 'GET' }, 6000);
    if (health.ok) {
      return { status: 'Connected' };
    }
    if (health.status === 404) {
      const detected = await detectTheriadEndpoint(baseUrl);
      return detected
        ? { status: 'Connected', detail: `Using ${detected}` }
        : { status: 'Endpoint missing', detail: 'No supported Theriad endpoint responded.' };
    }
    return { status: 'API error', detail: `${health.status} ${health.statusText}` };
  } catch (error) {
    return classifyConnectionError(error);
  }
}

export async function sendTheriadRequest(
  apiUrl: string,
  request: TheriadRequest,
  signal?: AbortSignal
): Promise<TheriadResponse> {
  const baseUrl = normalizeBaseUrl(apiUrl);
  const endpoint = await detectTheriadEndpoint(baseUrl, signal);
  if (!endpoint) {
    throw new Error('Endpoint missing: Theriad backend does not expose /api/chat/send, /api/execute, or /api/route.');
  }

  const payload = payloadForEndpoint(endpoint, request);
  const response = await fetchWithTimeout(
    `${baseUrl}${endpoint}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    },
    90000
  );

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`Theriad API error (${response.status}): ${body || response.statusText}`);
  }

  const data = await response.json();
  return normalizeTheriadResponse(data, endpoint);
}

async function detectTheriadEndpoint(baseUrl: string, signal?: AbortSignal): Promise<string | null> {
  for (const endpoint of ENDPOINTS) {
    try {
      const response = await fetchWithTimeout(
        `${baseUrl}${endpoint}`,
        {
          method: 'OPTIONS',
          signal,
        },
        4000
      );
      if (response.ok || response.status === 405 || response.status === 422) {
        return endpoint;
      }
      if (response.status === 404) {
        continue;
      }
    } catch {
      // Some FastAPI/CORS deployments do not support OPTIONS. Try a safe POST probe.
      try {
        const probe = await fetchWithTimeout(
          `${baseUrl}${endpoint}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(endpoint === '/api/execute' ? { prompt: 'health check' } : { messages: [{ role: 'user', content: 'health check' }] }),
            signal,
          },
          5000
        );
        if (probe.ok || probe.status === 422 || probe.status === 400) {
          return endpoint;
        }
      } catch {
        // Keep trying fallbacks.
      }
    }
  }
  return null;
}

function payloadForEndpoint(endpoint: string, request: TheriadRequest): unknown {
  if (endpoint === '/api/execute') {
    return {
      prompt: promptFromRequest(request),
      policy: 'balanced',
    };
  }
  if (endpoint === '/api/route') {
    return {
      prompt: promptFromRequest(request),
      browserContext: request.browserContext,
      runtimeProfile: request.runtimeProfile,
      client: request.client,
    };
  }
  return request;
}

function normalizeTheriadResponse(data: any, endpoint: string): TheriadResponse {
  if (endpoint === '/api/execute') {
    return {
      content: data.final_answer || data.content || '',
      model: data.model,
      provider: data.provider,
      routeTrace: data.trace,
      warnings: data.status && data.status !== 'completed' ? [`Execute status: ${data.status}`] : [],
    };
  }
  return {
    content: data.content || data.final_answer || data.message || '',
    model: data.model,
    provider: data.provider,
    routeTrace: data.routeTrace || data.trace,
    cost: data.cost,
    warnings: data.warnings || [],
    toolCalls: data.toolCalls || [],
  };
}

function promptFromRequest(request: TheriadRequest): string {
  const messages = request.messages.map((message) => `${message.role}: ${message.content}`).join('\n');
  const context = request.browserContext
    ? `\n\nBrowser context:\n${JSON.stringify(request.browserContext).slice(0, 16000)}`
    : '';
  return `${messages}${context}`;
}

function normalizeBaseUrl(apiUrl: string): string {
  return (apiUrl || 'http://localhost:8000').replace(/\/+$/, '');
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const timeoutController = new AbortController();
  const upstreamSignal = init.signal;
  const timeout = setTimeout(() => timeoutController.abort(), timeoutMs);
  const abortHandler = () => timeoutController.abort();
  upstreamSignal?.addEventListener('abort', abortHandler);
  try {
    return await fetch(url, { ...init, signal: timeoutController.signal });
  } finally {
    clearTimeout(timeout);
    upstreamSignal?.removeEventListener('abort', abortHandler);
  }
}

function classifyConnectionError(error: unknown): { status: TheriadConnectionStatus; detail?: string } {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
    return { status: 'Backend unavailable', detail: message };
  }
  if (message.includes('CORS')) {
    return { status: 'CORS/config error', detail: message };
  }
  if (message.includes('aborted')) {
    return { status: 'Backend unavailable', detail: 'Connection timed out.' };
  }
  return { status: 'API error', detail: message };
}
