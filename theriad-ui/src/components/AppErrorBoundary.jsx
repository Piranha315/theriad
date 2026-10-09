import React from 'react';
import { LoadingScreen } from './LoadingScreen.jsx';

export default class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    // Keep diagnostics available to developers without exposing a stack trace
    // or API jargon in the product UI.
    console.error('[Theriad UI] render boundary caught an error', error, info);
  }

  retry = () => {
    this.setState({ hasError: false });
  };

  reload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="theriad-error-screen" role="alert">
        <div className="theriad-error-card">
          <LoadingScreen
            variant="workspace"
            title="We hit a snag"
            message="This screen could not finish rendering, but your session is safe."
            delay={0}
          />
          <div className="theriad-error-actions">
            <button type="button" className="theriad-error-primary" onClick={this.retry}>
              Try this screen again
            </button>
            <button type="button" className="theriad-error-secondary" onClick={this.reload}>
              Reload Theriad
            </button>
          </div>
        </div>
      </div>
    );
  }
}
