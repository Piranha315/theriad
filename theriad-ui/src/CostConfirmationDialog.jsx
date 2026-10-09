import React, { useEffect, useRef } from 'react';

function money(value) {
  if (!Number.isFinite(value) || value <= 0) return 'less than a cent';
  if (value < 0.01) return 'less than a cent';
  return `about $${value.toFixed(value < 0.10 ? 2 : 1)}`;
}

export default function CostConfirmationDialog({ preview, onPaid, onFree, onCancel, allowQueueing = false }) {
  const paidRef = useRef(null);

  useEffect(() => {
    paidRef.current?.focus();
    const onKey = event => { if (event.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onCancel]);

  const source = Math.max(1, preview?.source_tokens ?? 1);
  const sent = Math.min(source, Math.max(1, preview?.sent_tokens ?? source));
  const compacted = Math.max(0, Math.round((1 - sent / source) * 100));

  return (
    <div className={`cost-guard-backdrop ${allowQueueing ? 'cost-guard-backdrop--queueable' : ''}`} role="presentation" onMouseDown={event => {
      if (event.target === event.currentTarget) onCancel();
    }}>
      <section className="cost-guard-dialog" role="dialog" aria-modal={allowQueueing ? 'false' : 'true'}
        aria-labelledby="cost-guard-title" aria-describedby="cost-guard-copy">
        <div className="cost-guard-kicker">Before Theriad runs this</div>
        <h2 id="cost-guard-title">This request carries a lot of context.</h2>
        <p id="cost-guard-copy">
          {preview?.reason || 'This request may use more context than a typical message.'}
          {' '}That is roughly {preview?.approximate_pages ?? 1} pages of material.
        </p>
        {allowQueueing && (
          <p className="cost-guard-queue-note">
            You can keep typing below. New prompts will wait in the task queue until this choice is made.
          </p>
        )}

        <div className="cost-guard-runway" aria-label={`${compacted}% of older context will be compacted`}>
          <div className="cost-guard-runway__topline">
            <span>Context load</span>
            <strong>{preview?.history_will_be_summarized ? 'Older chat condensed' : 'Large request'}</strong>
          </div>
          <div className="cost-guard-runway__track">
            <span style={{ width: `${Math.max(8, 100 - compacted)}%` }} />
          </div>
          <p>
            Recent messages stay intact. You can still retrieve an older conversation by name from Memory.
          </p>
        </div>

        <div className="cost-guard-estimate">
          <span>Selected route</span>
          <strong>{money(preview?.estimated_cost_usd)}</strong>
          <small>Rough estimate; final cost depends on the answer length.</small>
        </div>

        <div className="cost-guard-actions">
          <button ref={paidRef} type="button" className="cost-guard-primary" onClick={onPaid}>
            Use selected model
          </button>
          <button type="button" className="cost-guard-free" onClick={onFree}
            disabled={preview?.free_available === false}>
            Use a free model
            <span>{preview?.free_available === false
              ? 'Connect OpenRouter in Settings first'
              : 'May be slower or less capable'}</span>
          </button>
          <button type="button" className="cost-guard-cancel" onClick={onCancel}>Go back</button>
        </div>
      </section>
    </div>
  );
}
