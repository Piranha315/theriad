import React, { useEffect, useState } from 'react';
import { toFriendlyError } from './uiErrors.js';

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

function planPrice(offer, period) {
  if (!offer) return 'Contact us';
  return period === 'annual' ? offer.annual_price_display : offer.price_display;
}

export default function BillingSettings({ user, onOpenAuth }) {
  const [status, setStatus] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [period, setPeriod] = useState('monthly');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      apiFetch('/api/billing/status'),
      apiFetch('/api/pricing/plans'),
    ])
      .then(([nextStatus, nextCatalog]) => {
        if (cancelled) return;
        setStatus(nextStatus);
        setCatalog(nextCatalog);
      })
      .catch(nextError => {
        if (!cancelled) setError(toFriendlyError(nextError));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [user?.id]);

  async function startCheckout(planId) {
    if (busy || user?.is_guest) return;
    setBusy(planId);
    setError(null);
    try {
      const session = await apiFetch('/api/billing/checkout', {
        method: 'POST',
        headers: { 'Idempotency-Key': `theriad-${user.id}-${planId}-${period}` },
        body: JSON.stringify({ plan_id: planId, interval: period }),
      });
      if (!session?.url) throw new Error('Stripe did not return a checkout link.');
      window.location.assign(session.url);
    } catch (nextError) {
      setError(toFriendlyError(nextError));
    } finally {
      setBusy(null);
    }
  }

  async function openPortal() {
    if (busy || user?.is_guest) return;
    setBusy('portal');
    setError(null);
    try {
      const session = await apiFetch('/api/billing/portal', { method: 'POST' });
      if (!session?.url) throw new Error('Stripe did not return a billing link.');
      window.location.assign(session.url);
    } catch (nextError) {
      setError(toFriendlyError(nextError));
    } finally {
      setBusy(null);
    }
  }

  const offers = catalog?.offers?.filter(offer => offer.category === 'subscription') ?? [];
  const subscription = status?.subscription;
  const policyOnly = !status?.checkout_available;

  return (
    <section className="theriad-settings-card rounded-xl border border-line bg-surface-raised p-5 sm:p-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
        <div className="max-w-2xl">
          <h2 className="text-lg font-semibold text-ink">Plan &amp; billing</h2>
          <p className="mt-1 text-sm leading-relaxed text-ink-muted">
            Start free, connect your own provider, or choose a Theriad plan when billing is enabled for this deployment.
          </p>
        </div>
        {!loading && !user?.is_guest && subscription && status?.checkout_available && (
          <button
            type="button"
            onClick={openPortal}
            disabled={Boolean(busy)}
            className="rounded-lg border border-line px-3 py-2 text-xs font-medium text-ink-muted transition-colors hover:border-line-strong hover:bg-surface-hover hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy === 'portal' ? 'Opening...' : 'Manage billing'}
          </button>
        )}
      </div>

      {loading ? (
        <p className="mt-5 text-sm text-ink-faint">Checking plan availability...</p>
      ) : (
        <>
          <div className="mt-5 flex flex-col gap-3 rounded-lg border border-line bg-surface px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium text-ink">
                {subscription?.plan_id
                  ? `${subscription.plan_id[0].toUpperCase()}${subscription.plan_id.slice(1)} plan`
                  : user?.is_guest ? 'Free guest plan' : 'Free workspace'}
              </p>
              <p className="mt-1 text-xs text-ink-faint">
                {subscription?.status
                  ? `${subscription.status}${subscription.cancel_at_period_end ? ' · Ends at period close' : ''}`
                  : user?.is_guest
                  ? 'Sign up to keep this workspace and connect paid providers.'
                  : 'Use BYOK now; a subscription is optional until billing is configured.'}
              </p>
            </div>
            {user?.is_guest && (
              <button
                type="button"
                onClick={onOpenAuth}
                className="self-start rounded-lg border border-accent/40 bg-accent/10 px-3 py-2 text-xs font-semibold text-accent transition-colors hover:bg-accent/20"
              >
                Create an account
              </button>
            )}
          </div>

          {policyOnly ? (
            <p className="mt-4 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs leading-relaxed text-warn">
              Plans are currently policy-only on this server. No checkout or charge can be created until Stripe test-mode configuration is verified.
            </p>
          ) : (
            <>
              <div className="mt-5 flex items-center gap-2" aria-label="Billing period">
                {['monthly', 'annual'].map(option => (
                  <button
                    key={option}
                    type="button"
                    aria-pressed={period === option}
                    onClick={() => setPeriod(option)}
                    className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                      period === option
                        ? 'border-accent/50 bg-accent/10 text-accent'
                        : 'border-line text-ink-muted hover:border-line-strong hover:text-ink'
                    }`}
                  >
                    {option === 'annual' ? 'Annual' : 'Monthly'}
                  </button>
                ))}
              </div>
              <div className="mt-4 divide-y divide-line border-y border-line">
                {offers.map(offer => (
                  <div key={offer.id} className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-ink">{offer.name}</p>
                      <p className="mt-1 text-xs text-ink-muted">{offer.summary}</p>
                    </div>
                    <div className="flex items-center gap-3 sm:shrink-0">
                      <span className="text-sm font-medium text-ink">{planPrice(offer, period)}</span>
                      <button
                        type="button"
                        onClick={() => startCheckout(offer.id)}
                        disabled={Boolean(busy) || user?.is_guest}
                        className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-xs font-semibold text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {busy === offer.id ? 'Opening...' : 'Choose plan'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {error && (
        <p className="mt-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
