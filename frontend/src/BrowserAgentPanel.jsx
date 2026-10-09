import { useEffect, useRef, useState } from 'react';

const TERMINAL = new Set(['completed', 'failed', 'stopped', 'budget_exceeded']);
const POLL_DELAYS = { starting: 500, running: 1000, waiting_for_approval: 2500 };

async function request(path, init) {
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

export function BrowserAgentPanel() {
  const [goal, setGoal] = useState('');
  const [startUrl, setStartUrl] = useState('');
  const [session, setSession] = useState(null);
  const [error, setError] = useState('');
  const timer = useRef();

  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (!session?.session_id || TERMINAL.has(session.status)) return undefined;
    timer.current = setTimeout(async () => {
      try {
        const next = await request(`/api/browser/session/${encodeURIComponent(session.session_id)}`);
        setSession(next);
      } catch (pollError) { setError(pollError.message); }
    }, POLL_DELAYS[session.status] || 3000);
    return () => clearTimeout(timer.current);
  }, [session?.session_id, session?.status, session?.activity_log?.length]);

  async function start(event) {
    event.preventDefault();
    setError('');
    try {
      setSession(await request('/api/browser/session/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal, start_url: startUrl || null, max_cost_usd: 0.05 }),
      }));
    } catch (startError) { setError(startError.message); }
  }
  async function command(name) {
    if (!session) return;
    setError('');
    try { setSession(await request(`/api/browser/session/${session.session_id}/${name}`, { method: 'POST' })); }
    catch (commandError) { setError(commandError.message); }
  }

  return <section className="agent-panel" data-testid="browser-agent-panel">
    <header><p className="eyebrow">Theriad</p><h1>Browser Agent</h1><p>Policy-gated session preview. No live browser is claimed by this build.</p></header>
    <form onSubmit={start} className="composer">
      <label>Task<textarea required value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="What should the agent investigate?" /></label>
      <label>Starting URL <input value={startUrl} onChange={(e) => setStartUrl(e.target.value)} placeholder="https://example.com (optional)" /></label>
      <button type="submit" disabled={session && !TERMINAL.has(session.status)}>Start task</button>
    </form>
    {error && <p role="alert">{error}</p>}
    {session && <>
      <section className="live-card" aria-live="polite"><div><strong>{session.status.replaceAll('_', ' ')}</strong><button className="quiet" onClick={() => command('stop')} disabled={TERMINAL.has(session.status)}>Stop</button></div><dl><dt>Current URL</dt><dd>{session.current_url || 'Not opened yet'}</dd><dt>Page summary</dt><dd>{session.current_page_summary || 'Waiting for observation'}</dd></dl></section>
      {session.pending_approval && <section className="approval-sheet" data-testid="approval-sheet"><h2>Approval required</h2><p>{session.pending_approval.reason}</p><p><strong>Action:</strong> {session.pending_approval.proposed_action}</p><button onClick={() => command('approve')}>Approve</button><button className="quiet" onClick={() => command('deny')}>Deny</button>{session.captcha_detected && <button className="quiet" onClick={() => command('use-api-search')}>Use API search instead</button>}</section>}
      {session.final_result && <section className="result"><h2>Result</h2><p>{session.final_result}</p></section>}
      <details className="diagnostics"><summary>Developer diagnostics</summary><pre>{JSON.stringify({ activity: session.activity_log, cost_usd: session.cost_usd, model_calls: session.model_calls }, null, 2)}</pre></details>
    </>}
  </section>;
}
