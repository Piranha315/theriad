import React from 'react';
import { createRoot } from 'react-dom/client';
import { ChatAgentStatus } from './ChatAgentStatus.jsx';
import { ContextPanel } from './ContextPanel.jsx';
import { BrowserAgentPanel } from './BrowserAgentPanel.jsx';
import './app.css';

createRoot(document.getElementById('root')).render(
  <main className="app-shell">
    <BrowserAgentPanel />
    <ContextPanel />
  </main>
);
