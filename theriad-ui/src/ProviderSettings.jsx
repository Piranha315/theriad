import React, { useCallback, useEffect, useState } from 'react';
import { toFriendlyError } from './uiErrors.js';
import { useDelayedLoading } from './components/useDelayedLoading.js';

const CONNECTIONS = [
  {
    id: 'theriad',
    keyProvider: 'openrouter',
    name: 'Theriad Router',
    description: 'Route each request across Theriad\'s cost-aware model catalog.',
  },
  {
    id: 'openai',
    keyProvider: 'openai',
    name: 'OpenAI direct',
    description: 'Send assistant requests through your OpenAI API account.',
  },
  {
    id: 'anthropic',
    keyProvider: 'anthropic',
    name: 'Anthropic direct',
    description: 'Send assistant requests through your Anthropic API account.',
  },
];

const PROVIDERS = [
  {
    id: 'openrouter',
    name: 'OpenRouter',
    label: 'OpenRouter API key',
    placeholder: 'sk-or-v1-...',
    envName: 'OPENROUTER_API_KEY',
    help: 'Powers Theriad\'s cross-model routing and cost controls.',
  },
  {
    id: 'openai',
    name: 'OpenAI',
    label: 'OpenAI API key',
    placeholder: 'sk-...',
    envName: 'OPENAI_API_KEY',
    help: 'Uses API billing from your OpenAI Platform account. ChatGPT subscriptions are separate.',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    label: 'Anthropic API key',
    placeholder: 'sk-ant-...',
    envName: 'ANTHROPIC_API_KEY',
    help: 'Uses API billing from your Anthropic Console account.',
  },
];

async function apiFetch(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options;
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
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

function sourceLabel(source, storage) {
  if (source === 'env') return 'Environment';
  if (source === 'local_settings') {
    return storage === 'account_scoped_local_file' ? 'Saved for this account' : 'Saved locally';
  }
  return 'Not configured';
}

function ProviderSkeleton() {
  return (
    <div className="space-y-4" aria-label="Loading assistant connections">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {[0, 1, 2].map(item => (
          <div key={item} className="theriad-skeleton-block h-28 rounded-xl" />
        ))}
      </div>
      <div className="theriad-skeleton-block h-44 rounded-xl" />
    </div>
  );
}

export default function ProviderSettings() {
  const [providerData, setProviderData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [inputs, setInputs] = useState({});
  const [busy, setBusy] = useState(null);
  const [feedback, setFeedback] = useState({});
  const showSkeleton = useDelayedLoading(loading);
  const accountScopedStorage = providerData?.storage === 'account_scoped_local_file';

  const publishProviderData = useCallback(data => {
    setProviderData(data);
    window.dispatchEvent(new CustomEvent('theriad:providers-changed', { detail: data }));
  }, []);

  const loadProviders = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    setLoadError(null);
    try {
      const data = await apiFetch('/api/settings/providers');
      publishProviderData(data);
    } catch (error) {
      setLoadError(toFriendlyError(error));
    } finally {
      if (!quiet) setLoading(false);
    }
  }, [publishProviderData]);

  useEffect(() => {
    loadProviders();
  }, [loadProviders]);

  useEffect(() => {
    const refreshAfterOAuth = () => loadProviders({ quiet: true });
    window.addEventListener('focus', refreshAfterOAuth);
    return () => window.removeEventListener('focus', refreshAfterOAuth);
  }, [loadProviders]);

  function statusFor(providerId) {
    return providerData?.providers?.[providerId] ?? {
      configured: false,
      source: 'missing',
      masked_key: null,
    };
  }

  function updateFeedback(providerId, next) {
    setFeedback(current => ({ ...current, [providerId]: next }));
  }

  async function saveKey(providerId) {
    const apiKey = inputs[providerId]?.trim();
    if (!apiKey || busy) return;
    setBusy({ providerId, action: 'save' });
    updateFeedback(providerId, null);
    try {
      await apiFetch(`/api/settings/providers/${providerId}/key`, {
        method: 'POST',
        body: JSON.stringify({ api_key: apiKey }),
      });
      setInputs(current => ({ ...current, [providerId]: '' }));
      updateFeedback(providerId, { type: 'success', text: 'Key saved.' });
      await loadProviders({ quiet: true });
    } catch (error) {
      updateFeedback(providerId, { type: 'error', text: toFriendlyError(error) });
    } finally {
      setBusy(null);
    }
  }

  async function removeKey(providerId) {
    if (busy) return;
    setBusy({ providerId, action: 'remove' });
    updateFeedback(providerId, null);
    try {
      await apiFetch(`/api/settings/providers/${providerId}/key`, { method: 'DELETE' });
      setInputs(current => ({ ...current, [providerId]: '' }));
      updateFeedback(providerId, { type: 'success', text: 'Saved key removed.' });
      await loadProviders({ quiet: true });
    } catch (error) {
      updateFeedback(providerId, { type: 'error', text: toFriendlyError(error) });
    } finally {
      setBusy(null);
    }
  }

  async function testProvider(providerId) {
    if (busy) return;
    setBusy({ providerId, action: 'test' });
    updateFeedback(providerId, null);
    try {
      const result = await apiFetch(`/api/settings/providers/${providerId}/test`, { method: 'POST' });
      const passed = result?.ok !== false && result?.status !== 'error';
      updateFeedback(providerId, {
        type: passed ? 'success' : 'error',
        text: result?.message ?? (passed ? 'Connection verified.' : 'Connection test failed.'),
      });
    } catch (error) {
      updateFeedback(providerId, { type: 'error', text: toFriendlyError(error) });
    } finally {
      setBusy(null);
    }
  }

  async function connectOpenRouter() {
    if (busy) return;
    setBusy({ providerId: 'openrouter', action: 'oauth' });
    updateFeedback('openrouter', null);
    try {
      const result = await apiFetch('/api/settings/providers/openrouter/oauth/start', {
        method: 'POST',
      });
      const popup = window.open(
        result.authorization_url,
        'theriad-openrouter-connect',
        'popup,width=560,height=760,noopener',
      );
      if (!popup) throw new Error('Your browser blocked the OpenRouter window. Allow popups and try again.');
      updateFeedback('openrouter', {
        type: 'success',
        text: 'Finish connecting in the OpenRouter window. Theriad will refresh when you return.',
      });
    } catch (error) {
      updateFeedback('openrouter', { type: 'error', text: toFriendlyError(error) });
    } finally {
      setBusy(null);
    }
  }

  async function setActiveProvider(connectionId) {
    const connection = CONNECTIONS.find(item => item.id === connectionId);
    if (!connection || !statusFor(connection.keyProvider).configured || busy) return;
    setBusy({ providerId: connectionId, action: 'activate' });
    setLoadError(null);
    try {
      const result = await apiFetch('/api/settings/providers/active', {
        method: 'PUT',
        body: JSON.stringify({ provider: connectionId }),
      });
      const next = result?.providers
        ? result
        : { ...providerData, active_provider: result?.active_provider ?? connectionId };
      publishProviderData(next);
    } catch (error) {
      setLoadError(toFriendlyError(error));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="theriad-settings-card rounded-xl border border-line bg-surface-raised p-5 sm:p-6">
      <div className="mb-5 max-w-2xl">
        <h2 className="text-lg font-semibold text-ink">Assistant connections</h2>
        <p className="mt-1 text-sm leading-relaxed text-ink-muted">
          Choose how Theriad handles chat and model-backed requests. Keys are kept out of browser storage,
          scoped to this account when account storage is available, and never shown again in full.
        </p>
        {accountScopedStorage && (
          <p className="mt-3 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs leading-relaxed text-warn">
            This deployment is using account-scoped local storage. Move provider secrets to a managed secret store before opening the workspace to multiple users.
          </p>
        )}
      </div>

      {showSkeleton && loading ? <ProviderSkeleton /> : loading ? null : loadError && !providerData ? (
        <div className="rounded-xl border border-danger/30 bg-danger/10 p-4">
          <p className="text-sm font-medium text-danger">Could not load assistant connections.</p>
          <p className="mt-1 break-words text-xs text-danger/80">{loadError}</p>
          <button
            type="button"
            onClick={() => loadProviders()}
            className="mt-3 rounded-lg border border-danger/30 px-3 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10"
          >
            Try again
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          <fieldset>
            <legend className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-faint">
              Active assistant
            </legend>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {CONNECTIONS.map(connection => {
                const providerStatus = statusFor(connection.keyProvider);
                const selected = providerData?.active_provider === connection.id;
                const activating = busy?.providerId === connection.id && busy?.action === 'activate';
                return (
                  <button
                    key={connection.id}
                    type="button"
                    aria-pressed={selected}
                    disabled={!providerStatus.configured || Boolean(busy)}
                    onClick={() => setActiveProvider(connection.id)}
                    className={`min-h-28 rounded-xl border p-4 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:cursor-not-allowed ${
                      selected
                        ? 'border-accent/60 bg-accent/10'
                        : providerStatus.configured
                        ? 'border-line bg-surface hover:border-line-strong hover:bg-surface-hover'
                        : 'border-line bg-surface opacity-60'
                    }`}
                  >
                    <span className="flex items-center justify-between gap-3">
                      <span className={`text-sm font-semibold ${selected ? 'text-accent' : 'text-ink'}`}>
                        {connection.name}
                      </span>
                      <span className={`h-2 w-2 flex-shrink-0 rounded-full ${providerStatus.configured ? 'bg-ok' : 'bg-warn'}`} />
                    </span>
                    <span className="mt-2 block text-xs leading-relaxed text-ink-muted">
                      {connection.description}
                    </span>
                    <span className={`mt-2 block text-[11px] ${selected ? 'text-accent' : 'text-ink-faint'}`}>
                      {activating ? 'Switching...' : selected ? 'Active' : providerStatus.configured ? 'Ready' : 'Add a key below'}
                    </span>
                  </button>
                );
              })}
            </div>
          </fieldset>

          {loadError && (
            <p className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">
              {loadError}
            </p>
          )}

          <div className="divide-y divide-line border-y border-line">
            {PROVIDERS.map(provider => {
              const providerStatus = statusFor(provider.id);
              const isEnvManaged = providerStatus.source === 'env';
              const providerBusy = busy?.providerId === provider.id;
              const providerFeedback = feedback[provider.id];
              return (
                <div key={provider.id} className="py-5">
                  <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold text-ink">{provider.name}</h3>
                        <span className={`rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${
                          providerStatus.configured
                            ? 'border-ok/30 bg-ok/10 text-ok'
                            : 'border-warn/30 bg-warn/10 text-warn'
                        }`}>
                          {providerStatus.configured ? 'Configured' : 'Key needed'}
                        </span>
                      </div>
                      <p className="mt-1 max-w-xl text-xs leading-relaxed text-ink-muted">{provider.help}</p>
                      {providerStatus.configured && (
                        <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-faint">
                          <span>{sourceLabel(providerStatus.source, providerData?.storage)}</span>
                          {providerStatus.masked_key && <span className="font-mono">{providerStatus.masked_key}</span>}
                        </p>
                      )}
                    </div>
                    {providerStatus.configured && (
                      <button
                        type="button"
                        onClick={() => testProvider(provider.id)}
                        disabled={Boolean(busy)}
                        className="self-start rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink-muted transition-colors hover:border-line-strong hover:bg-surface-hover hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {providerBusy && busy?.action === 'test' ? 'Testing...' : 'Test'}
                      </button>
                    )}
                  </div>

                  {isEnvManaged ? (
                    <details className="mt-3 rounded-lg border border-line bg-surface px-3 py-2">
                      <summary className="cursor-pointer text-xs text-ink-muted">Managed by environment</summary>
                      <p className="mt-2 text-xs leading-relaxed text-ink-faint">
                        This key comes from <span className="font-mono">{provider.envName}</span>. Remove that environment variable to replace it here.
                      </p>
                    </details>
                  ) : (
                    <div className="mt-3 space-y-3">
                      {provider.id === 'openrouter' && (
                        <div className="flex flex-wrap items-center gap-3">
                          <button
                            type="button"
                            onClick={connectOpenRouter}
                            disabled={Boolean(busy)}
                            className="rounded-lg border border-accent/40 bg-accent/15 px-4 py-2 text-sm font-semibold text-accent transition-colors hover:bg-accent/25 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {providerBusy && busy?.action === 'oauth'
                              ? 'Opening OpenRouter...'
                              : providerStatus.configured
                              ? 'Reconnect with OpenRouter'
                              : 'Connect with OpenRouter'}
                          </button>
                          <span className="text-xs text-ink-faint">One account, no key copying.</span>
                        </div>
                      )}
                      <div className="flex flex-col gap-2 sm:flex-row">
                        <label className="sr-only" htmlFor={`provider-key-${provider.id}`}>{provider.label}</label>
                        <input
                          id={`provider-key-${provider.id}`}
                          type="password"
                          autoComplete="off"
                          spellCheck={false}
                          value={inputs[provider.id] ?? ''}
                          onChange={event => {
                            setInputs(current => ({ ...current, [provider.id]: event.target.value }));
                            updateFeedback(provider.id, null);
                          }}
                          onKeyDown={event => {
                            if (event.key === 'Enter') saveKey(provider.id);
                          }}
                          placeholder={providerStatus.configured ? `Paste a replacement ${provider.placeholder}` : provider.placeholder}
                          className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-3 py-2 font-mono text-sm text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/60 focus:ring-1 focus:ring-accent/30"
                        />
                        <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => saveKey(provider.id)}
                          disabled={!inputs[provider.id]?.trim() || Boolean(busy)}
                          className="rounded-lg border border-accent/30 bg-accent/10 px-4 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {providerBusy && busy?.action === 'save' ? 'Saving...' : providerStatus.configured ? 'Replace' : 'Save'}
                        </button>
                        {providerStatus.configured && providerStatus.source === 'local_settings' && (
                          <button
                            type="button"
                            onClick={() => removeKey(provider.id)}
                            disabled={Boolean(busy)}
                            className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger/20 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {providerBusy && busy?.action === 'remove' ? 'Removing...' : 'Remove'}
                          </button>
                        )}
                        </div>
                      </div>
                    </div>
                  )}

                  {providerFeedback && (
                    <p
                      className={`mt-2 break-words text-xs ${providerFeedback.type === 'success' ? 'text-ok' : 'text-danger'}`}
                      role={providerFeedback.type === 'error' ? 'alert' : 'status'}
                    >
                      {providerFeedback.text}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
