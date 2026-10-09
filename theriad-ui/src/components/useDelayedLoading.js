import { useEffect, useState } from 'react';

// Fast requests should render directly into their final state. This delay is
// long enough to prevent a loader flash without making a slow page feel idle.
export const DEFAULT_LOADING_DELAY_MS = 200;

export function useDelayedLoading(isLoading, delay = DEFAULT_LOADING_DELAY_MS) {
  const [showLoading, setShowLoading] = useState(
    () => Boolean(isLoading) && delay <= 0,
  );

  useEffect(() => {
    if (!isLoading) {
      setShowLoading(false);
      return undefined;
    }

    if (delay <= 0) {
      setShowLoading(true);
      return undefined;
    }

    const timer = window.setTimeout(() => setShowLoading(true), delay);
    return () => window.clearTimeout(timer);
  }, [delay, isLoading]);

  return showLoading;
}
