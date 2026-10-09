import { DEFAULT_LOADING_DELAY_MS, useDelayedLoading } from './useDelayedLoading.js';

export function LoadingScreen({
  variant = 'workspace',
  title,
  message,
  delay = DEFAULT_LOADING_DELAY_MS,
}) {
  const isLanding = variant === 'landing';
  const resolvedTitle = title ?? (isLanding ? 'Loading Theriad' : 'Preparing your workspace');
  const resolvedMessage = message ?? (isLanding
    ? 'Bringing the cost-aware AI workflow into view…'
    : 'Restoring your session and loading the workspace…');
  const showLoading = useDelayedLoading(true, delay);

  if (!showLoading) {
    return <span className="sr-only" role="status" aria-live="polite">{resolvedMessage}</span>;
  }

  return (
    <div className="theriad-loading-screen" role="status" aria-live="polite">
      <div className="theriad-loading-header">
        <div className="theriad-loading-mark" aria-hidden="true">T</div>
        <div className="min-w-0">
          <p className="theriad-loading-brand">Theriad</p>
          <p className="theriad-loading-title">{resolvedTitle}</p>
          <p className="theriad-loading-message">{resolvedMessage}</p>
        </div>
      </div>

      <div className="theriad-loading-skeleton" aria-hidden="true">
        <span className="theriad-loading-line theriad-loading-line--wide" />
        <span className="theriad-loading-line" />
        <span className="theriad-loading-line theriad-loading-line--short" />
        <div className="theriad-loading-cards">
          <span />
          <span />
          <span />
        </div>
      </div>

      <span className="sr-only">{resolvedMessage}</span>
    </div>
  );
}
export function PageLoadingScreen({ label = 'Loading this view…' }) {
  return (
    <LoadingScreen
      variant="workspace"
      title="Loading view"
      message={label}
    />
  );
}
