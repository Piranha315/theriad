import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { DEFAULT_SETTINGS, type Settings } from './types';
import { testTheriadConnection, type TheriadConnectionStatus } from './theriadClient';

const PROVIDER_MODELS = {
  google: [
    { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', description: '1M token context' },
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', description: 'Fast and efficient' },
    { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite', description: 'Optimized for speed' },
  ],
};

function mergeSettings(value?: Partial<Settings>): Settings {
  return { ...DEFAULT_SETTINGS, ...(value || {}) };
}

function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [showGoogleKey, setShowGoogleKey] = useState(false);
  const [showComposioKey, setShowComposioKey] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<TheriadConnectionStatus | 'Not tested'>('Not tested');
  const [connectionDetail, setConnectionDetail] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const [bridgeStatus, setBridgeStatus] = useState('Not paired');

  useEffect(() => {
    chrome.storage.local.get(['theriadSettings'], (result) => {
      setSettings(mergeSettings(result.theriadSettings));
    });
  }, []);

  const handleSave = () => {
    chrome.storage.local.set({ theriadSettings: settings }, () => {
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);

      chrome.runtime.sendMessage({ type: 'SETTINGS_UPDATED' }, () => {
        if (chrome.runtime.lastError) {
          console.log('Sidebar not active, but settings saved');
        }
      });
    });
  };

  const handleTestConnection = async () => {
    setIsTesting(true);
    setConnectionStatus('Not tested');
    setConnectionDetail('');
    const result = await testTheriadConnection(settings.theriadApiUrl);
    setConnectionStatus(result.status);
    setConnectionDetail(result.detail || '');
    setIsTesting(false);
  };

  const handlePairCurrentTab = async () => {
    setBridgeStatus('Requesting browser access…');
    const granted = await chrome.permissions.request({ origins: ['<all_urls>'] });
    if (!granted) {
      setBridgeStatus('Browser access was not granted.');
      return;
    }
    const response = await chrome.runtime.sendMessage({
      type: 'PAIR_AGENT_BRIDGE',
      apiUrl: settings.theriadApiUrl,
      pairingCode: settings.agentBridgePairingCode,
    });
    setBridgeStatus(response?.success ? 'Paired to current tab' : `Pairing failed — ${response?.error || 'unknown error'}`);
    if (response?.success) {
      const updated = { ...settings, agentBridgeEnabled: true, agentBridgePairingCode: '' };
      setSettings(updated);
      chrome.storage.local.set({ theriadSettings: updated });
    }
  };

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h1>Theriad Browser Settings</h1>
        <p>Configure the Theriad backend first. Google and Composio are optional feature modes.</p>
      </div>

      <div className="settings-content">
        <section className="settings-section">
          <h2>Theriad Router Chat</h2>
          <div className="setting-group">
            <label>Theriad Backend URL</label>
            <input
              value={settings.theriadApiUrl}
              onChange={(event) => setSettings({ ...settings, theriadApiUrl: event.target.value })}
              placeholder="http://localhost:8000"
              className="api-key-input"
            />
            <p className="help-text">Normal chat and page-aware analysis route through this backend. No Google API key is required.</p>
          </div>

          <div className="setting-group">
            <label>Theriad Mode</label>
            <select
              value={settings.theriadMode}
              onChange={(event) => setSettings({ ...settings, theriadMode: event.target.value as Settings['theriadMode'] })}
              className="model-select"
            >
              <option value="chat">Chat endpoint</option>
              <option value="execute">Execute endpoint fallback</option>
              <option value="router">Pure router fallback</option>
            </select>
          </div>

          <div className="setting-group">
            <label>Runtime Profile</label>
            <select
              value={settings.runtimeProfile}
              onChange={(event) => setSettings({ ...settings, runtimeProfile: event.target.value as Settings['runtimeProfile'] })}
              className="model-select"
            >
              <option value="consumer_secure">Secure</option>
              <option value="personal_trusted">Personal trusted</option>
            </select>
            <p className="help-text">Secure mode asks for confirmation around a broader set of sensitive actions.</p>
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.includePageContext}
              onChange={(event) => setSettings({ ...settings, includePageContext: event.target.checked })}
            />
            Include current page context by default
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.includeBrowserMemory}
              onChange={(event) => setSettings({ ...settings, includeBrowserMemory: event.target.checked })}
            />
            Include browser memory summaries
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.showRouteTrace}
              onChange={(event) => setSettings({ ...settings, showRouteTrace: event.target.checked })}
            />
            Show route trace under assistant replies
          </label>

          <button className="secondary-button" onClick={handleTestConnection} disabled={!settings.theriadApiUrl || isTesting}>
            {isTesting ? 'Testing…' : 'Test Theriad Connection'}
          </button>
          <p className={`connection-status ${connectionStatus.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`}>
            {connectionStatus}
            {connectionDetail ? ` — ${connectionDetail}` : ''}
          </p>
        </section>

        <section className="settings-section">
          <h2>Local Agent Bridge</h2>
          <p className="help-text">
            Pair only the current tab to a local Theriad task. The agent cannot use unrelated tabs unless you explicitly pair them.
          </p>
          <div className="setting-group">
            <label>One-time pairing code</label>
            <input
              value={settings.agentBridgePairingCode || ''}
              onChange={(event) => setSettings({ ...settings, agentBridgePairingCode: event.target.value.toUpperCase() })}
              placeholder="AB12-CD34"
              className="api-key-input"
            />
          </div>
          <button
            className="secondary-button"
            onClick={handlePairCurrentTab}
            disabled={!settings.theriadApiUrl || !settings.agentBridgePairingCode}
          >
            Pair current tab
          </button>
          <p className="connection-status">{bridgeStatus}</p>
        </section>

        <section className="settings-section">
          <h2>Browser Tools / Computer Use</h2>
          <p className="help-text">Optional Gemini Computer Use mode for visual browser automation. Normal chat does not use this key.</p>
          <div className="setting-group">
            <label>Google Model</label>
            <select
              value={settings.model}
              onChange={(event) => setSettings({ ...settings, model: event.target.value })}
              className="model-select"
            >
              {PROVIDER_MODELS.google.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} - {model.description}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-group">
            <label>Google API Key</label>
            <div className="api-key-input-wrapper">
              <input
                type={showGoogleKey ? 'text' : 'password'}
                value={settings.apiKey}
                onChange={(event) => setSettings({ ...settings, apiKey: event.target.value, provider: 'theriad' })}
                placeholder="Only required for Browser Tools"
                className="api-key-input"
              />
              <button type="button" className="toggle-visibility" onClick={() => setShowGoogleKey(!showGoogleKey)}>
                {showGoogleKey ? '👁️' : '👁️‍🗨️'}
              </button>
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h2>Tool Integrations</h2>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={Boolean(settings.enableComposioTools)}
              onChange={(event) => setSettings({ ...settings, enableComposioTools: event.target.checked })}
            />
            Enable Composio Tool Router
          </label>
          <div className="setting-group">
            <label>Composio API Key</label>
            <div className="api-key-input-wrapper">
              <input
                type={showComposioKey ? 'text' : 'password'}
                value={settings.composioApiKey || ''}
                onChange={(event) => setSettings({ ...settings, composioApiKey: event.target.value })}
                placeholder="Only required for Composio Tool Router"
                className="api-key-input"
              />
              <button type="button" className="toggle-visibility" onClick={() => setShowComposioKey(!showComposioKey)}>
                {showComposioKey ? '👁️' : '👁️‍🗨️'}
              </button>
            </div>
          </div>
        </section>

        <button className={`save-button ${saved ? 'saved' : ''}`} onClick={handleSave} disabled={!settings.theriadApiUrl}>
          {saved ? '✓ Saved!' : 'Save Settings'}
        </button>

        <div className="info-box">
          <h3>🔒 Privacy & Security</h3>
          <p>
            Theriad receives only the active chat and capped page context. Google and Composio keys stay in browser storage and are only used when their optional modes are enabled.
          </p>
        </div>
      </div>
    </div>
  );
}

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<SettingsPage />);
