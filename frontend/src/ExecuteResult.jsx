import { AgentDebugPanel } from './AgentDebugPanel';

const MISSING_ANSWER = 'Execution completed, but no final answer was generated.';

export function ExecuteResult({ response }) {
  if (!response) return null;

  const finalAnswer = response.final_answer?.trim() || MISSING_ANSWER;

  return (
    <section className="execute-result" data-testid="execute-result">
      <div className="execute-answer" aria-live="polite">
        <h2>Answer</h2>
        <p data-testid="execute-final-answer">{finalAnswer}</p>
      </div>

      <details className="execution-details">
        <summary>Execution details</summary>
        <AgentDebugPanel trace={response.trace || []} />
        {response.artifacts?.length ? (
          <details>
            <summary>Collected artifacts</summary>
            <pre>{JSON.stringify(response.artifacts, null, 2)}</pre>
          </details>
        ) : null}
      </details>
    </section>
  );
}
