import { DEFAULT_LOADING_DELAY_MS, useDelayedLoading } from './useDelayedLoading.js';

export function DelayedLoader({
  loading,
  label = 'Loading…',
  delay = DEFAULT_LOADING_DELAY_MS,
  className = '',
}) {
  const showLoading = useDelayedLoading(loading, delay);
  if (!showLoading) return null;

  return (
    <div className={`theriad-loading-indicator ${className}`.trim()} role="status" aria-live="polite">
      <span className="theriad-loading-signal" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>{label}</span>
    </div>
  );
}
