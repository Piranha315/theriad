export const AGENT_STATUSES = [
  'idle',
  'starting',
  'running',
  'waiting_for_approval',
  'failed',
  'completed',
  'stopped',
  'budget_exceeded',
];

const statusLabel = (status) => (status || 'starting').replaceAll('_', ' ');

export function ChatAgentStatus({ message }) {
  if (!message?.agent_run_id) return null;
  const trace = message.trace || [];
  return (
    <section className="agent-progress" data-testid="chat-agent-status">
      <header>
        <strong>{statusLabel(message.status)}</strong>
        <span>{message.agent_run_id}</span>
      </header>
      <details>
        <summary>Agent trace</summary>
        <ul>
          {trace.map((step, index) => (
            <li key={index}>
              <code>{step.tool || step.status}</code>
              {step.command ? <span> command: {step.command}</span> : null}
              {step.url ? <span> url: {step.url}</span> : null}
              {step.error ? <span> error: {step.error}</span> : null}
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
