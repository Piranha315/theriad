export function AgentDebugPanel({ trace = [] }) {
  return (
    <main data-testid="agent-debug-panel">
      <h1>Agent Debug Trace</h1>
      <pre>{JSON.stringify(trace, null, 2)}</pre>
    </main>
  );
}
