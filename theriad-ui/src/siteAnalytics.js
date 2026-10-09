import { useEffect } from 'react';

const ENDPOINT = '/api/analytics/collect';
const SESSION_KEY = 'theriad_site_analytics_session';
const SESSION_IDLE_MS = 30 * 60 * 1000;
const HEARTBEAT_MS = 30 * 1000;

function makeId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function currentPath() {
  return `${window.location.pathname}${window.location.hash}` || '/';
}

function readSession() {
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem(SESSION_KEY) || 'null'); } catch { /* ignore */ }
  const now = Date.now();
  if (!saved?.id || now - Number(saved.lastActivityAt || 0) > SESSION_IDLE_MS) {
    saved = { id: makeId(), lastActivityAt: now };
  } else {
    saved.lastActivityAt = now;
  }
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(saved)); } catch { /* ignore */ }
  return saved;
}

function markActive(session) {
  session.lastActivityAt = Date.now();
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(session)); } catch { /* ignore */ }
}

function send(event, session, useBeacon = false) {
  const payload = JSON.stringify({
    event,
    session_id: session.id,
    path: currentPath(),
    referrer: document.referrer || null,
  });

  if (useBeacon && navigator.sendBeacon) {
    const accepted = navigator.sendBeacon(
      ENDPOINT,
      new Blob([payload], { type: 'application/json' }),
    );
    if (accepted) return;
  }

  fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload,
    keepalive: useBeacon,
  }).catch(() => {});
}

/** Track SPA page views and visible activity for anonymous site analytics. */
export function useSiteAnalytics() {
  // The tracker deliberately runs outside auth: landing-page visitors have no
  // Theriad account yet, and traffic measurement must not link to chat data.
  useEffect(() => {
    let teardown = () => {};
    const timer = window.setTimeout(() => {
      const session = readSession();
      send('page_view', session);

      const heartbeat = window.setInterval(() => {
        if (document.visibilityState === 'visible') {
          markActive(session);
          send('heartbeat', session);
        }
      }, HEARTBEAT_MS);

      const onHashChange = () => {
        if (Date.now() - session.lastActivityAt > SESSION_IDLE_MS) {
          session.id = makeId();
        }
        markActive(session);
        send('page_view', session);
      };
      const onPageHide = () => send('session_end', session, true);
      const onVisibilityChange = () => {
        if (document.visibilityState === 'visible') {
          markActive(session);
          send('heartbeat', session);
        }
      };

      window.addEventListener('hashchange', onHashChange);
      window.addEventListener('pagehide', onPageHide);
      document.addEventListener('visibilitychange', onVisibilityChange);
      teardown = () => {
        window.clearInterval(heartbeat);
        window.removeEventListener('hashchange', onHashChange);
        window.removeEventListener('pagehide', onPageHide);
        document.removeEventListener('visibilitychange', onVisibilityChange);
      };
    }, 0);

    return () => {
      window.clearTimeout(timer);
      teardown();
    };
  }, []);
}
