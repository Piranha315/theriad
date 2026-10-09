// Background service worker for the extension

// Memory store for browser context
interface BrowserMemory {
  recentPages: Array<{ url: string; title: string; timestamp: number; context?: any }>;
  userPreferences: Record<string, any>;
  sessionData: Record<string, any>;
}

const memory: BrowserMemory = {
  recentPages: [],
  userPreferences: {},
  sessionData: {}
};

type BridgeCommand = {
  type: 'command';
  request_id: string;
  sequence: number;
  session_token: string;
  task_id: string;
  tab_id: number;
  tool: string;
  arguments: Record<string, any>;
};

let bridgeSocket: WebSocket | null = null;
let bridgeSessionToken = '';
let bridgeLastSequence = 0;
let bridgeHeartbeat: ReturnType<typeof setInterval> | null = null;

function bridgeUrl(apiUrl: string, pairingCode: string): string {
  const base = (apiUrl || 'http://localhost:8000').replace(/\/+$/, '');
  return `${base.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')}/api/agent/extension/ws?code=${encodeURIComponent(pairingCode)}`;
}

async function ensureBridgeContentScript(tabId: number): Promise<void> {
  try {
    await chrome.tabs.sendMessage(tabId, { type: 'PING' });
  } catch {
    await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
    await new Promise(resolve => setTimeout(resolve, 150));
  }
}

async function captureRedactedTab(tabId: number): Promise<string> {
  const tab = await chrome.tabs.get(tabId);
  if (tab.windowId === undefined) throw new Error('Assigned tab has no window.');
  await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      const sensitive = document.querySelectorAll(
        'input[type="password"], input[autocomplete*="cc-"], input[autocomplete="one-time-code"]'
      );
      sensitive.forEach((element) => {
        const node = element as HTMLElement;
        node.dataset.theriadPreviousFilter = node.style.filter;
        node.style.filter = 'blur(12px)';
      });
    },
  });
  try {
    return await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png', quality: 80 });
  } finally {
    await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        document.querySelectorAll('[data-theriad-previous-filter]').forEach((element) => {
          const node = element as HTMLElement;
          node.style.filter = node.dataset.theriadPreviousFilter || '';
          delete node.dataset.theriadPreviousFilter;
        });
      },
    });
  }
}

async function executeBridgeCommand(command: BridgeCommand): Promise<any> {
  if (command.sequence <= bridgeLastSequence) throw new Error('Rejected replayed or out-of-order command.');
  bridgeLastSequence = command.sequence;
  const tabId = command.tab_id;
  const args = command.arguments || {};
  if (command.tool === 'observe') {
    await ensureBridgeContentScript(tabId);
    return chrome.tabs.sendMessage(tabId, { type: 'GET_PAGE_CONTEXT' });
  }
  if (command.tool === 'screenshot') {
    return { screenshot: await captureRedactedTab(tabId), redacted_sensitive_fields: true };
  }
  if (command.tool === 'navigate') {
    const url = String(args.url || '');
    const parsed = new URL(url);
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Only HTTP(S) navigation is allowed.');
    await chrome.tabs.update(tabId, { url: parsed.toString() });
    return { url: parsed.toString() };
  }
  if (command.tool === 'new_tab') {
    const created = await chrome.tabs.create({ url: args.url || 'about:blank', active: true });
    if (!created.id) throw new Error('Chrome did not return a tab ID.');
    return { tab_id: created.id, url: created.url };
  }
  if (command.tool === 'close_tab') {
    await chrome.tabs.remove(tabId);
    return { closed: true };
  }
  if (command.tool === 'select_tab') {
    await chrome.tabs.update(tabId, { active: true });
    return { selected: true };
  }
  await ensureBridgeContentScript(tabId);
  const actionMap: Record<string, string> = {
    click: 'click', fill: 'fill', type: 'keyboard_type', keypress: 'press_key',
    scroll: 'scroll', drag: 'drag_drop', hover: 'hover', mouse_move: 'mouse_move', clear: 'clear_input',
  };
  const action = actionMap[command.tool];
  if (!action) throw new Error(`Unsupported extension tool: ${command.tool}`);
  const result = await chrome.tabs.sendMessage(tabId, {
    type: 'EXECUTE_ACTION', action,
    target: args.target, selector: args.selector, value: args.value ?? args.text,
    key: args.key, keys: args.keys, coordinates: args.coordinates,
    destination: args.destination, direction: args.direction, amount: args.amount,
  });
  if (result?.success === false) throw new Error(result.message || 'Browser action failed.');
  return result;
}

async function pairAgentBridge(apiUrl: string, pairingCode: string): Promise<{ success: boolean; error?: string }> {
  if (bridgeSocket) bridgeSocket.close();
  if (bridgeHeartbeat) clearInterval(bridgeHeartbeat);
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const assignedTab = tabs[0];
  if (!assignedTab?.id) return { success: false, error: 'No active tab to assign.' };
  return new Promise((resolve) => {
    const socket = new WebSocket(bridgeUrl(apiUrl, pairingCode));
    bridgeSocket = socket;
    const timer = setTimeout(() => resolve({ success: false, error: 'Pairing timed out.' }), 8000);
    socket.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'paired') {
        bridgeSessionToken = message.session_token;
        bridgeLastSequence = 0;
        bridgeHeartbeat = setInterval(() => {
          if (socket.readyState === WebSocket.OPEN && bridgeSessionToken) {
            socket.send(JSON.stringify({ type: 'heartbeat', session_token: bridgeSessionToken }));
          }
        }, 20_000);
        socket.send(JSON.stringify({ type: 'hello', tab_id: assignedTab.id }));
        return;
      }
      if (message.type === 'ready') {
        clearTimeout(timer);
        chrome.storage.local.set({ agentBridgeTaskId: message.task_id, agentBridgeTabId: assignedTab.id });
        resolve({ success: true });
        return;
      }
      if (message.type === 'command') {
        const command = message as BridgeCommand;
        try {
          const result = await executeBridgeCommand(command);
          socket.send(JSON.stringify({
            type: 'result', request_id: command.request_id, session_token: bridgeSessionToken,
            sequence: command.sequence, ok: true, result,
          }));
        } catch (error) {
          socket.send(JSON.stringify({
            type: 'result', request_id: command.request_id, session_token: bridgeSessionToken,
            sequence: command.sequence, ok: false, error: error instanceof Error ? error.message : String(error),
          }));
        }
      }
    };
    socket.onerror = () => {
      clearTimeout(timer);
      resolve({ success: false, error: 'Could not connect to the local Theriad bridge.' });
    };
    socket.onclose = () => {
      if (bridgeHeartbeat) clearInterval(bridgeHeartbeat);
      bridgeHeartbeat = null;
      bridgeSocket = null;
      bridgeSessionToken = '';
    };
  });
}

chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error: Error) => console.error(error));

// Listen for extension icon clicks
chrome.action.onClicked.addListener((tab) => {
  if (tab.id) {
    chrome.sidePanel.open({ tabId: tab.id });
  }
});

// Add page to memory
function addToMemory(page: { url: string; title: string; timestamp: number }) {
  memory.recentPages.unshift(page);
  if (memory.recentPages.length > 100) {
    memory.recentPages.pop();
  }

  // Save to chrome.storage for persistence
  chrome.storage.local.set({ browserMemory: memory });
}

// Load memory from storage on startup
chrome.storage.local.get('browserMemory', (result) => {
  if (result.browserMemory) {
    Object.assign(memory, result.browserMemory);
  }
});

// Listen for messages from the sidebar and content scripts
chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request.type === 'PAIR_AGENT_BRIDGE') {
    pairAgentBridge(request.apiUrl, request.pairingCode).then(sendResponse);
    return true;
  }
  // Get current tab info
  if (request.type === 'GET_TAB_INFO') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        sendResponse({
          url: tabs[0].url,
          title: tabs[0].title,
          id: tabs[0].id
        });
      }
    });
    return true;
  }

  // Get browser history
  if (request.type === 'GET_HISTORY') {
    const query = request.query || '';
    const maxResults = request.maxResults || 100;
    const startTime = request.startTime || Date.now() - (7 * 24 * 60 * 60 * 1000); // Last 7 days

    chrome.history.search({
      text: query,
      maxResults,
      startTime
    }, (results) => {
      sendResponse({ history: results });
    });
    return true;
  }

  // Get browser memory
  if (request.type === 'GET_MEMORY') {
    sendResponse({ memory });
    return true;
  }

  // Get page context from content script
  // Helper function to ensure content script is injected
  async function ensureContentScript(tabId: number): Promise<void> {
    try {
      // Try to ping the content script
      await chrome.tabs.sendMessage(tabId, { type: 'PING' });
    } catch (error) {
      try {
        await chrome.scripting.executeScript({
          target: { tabId },
          files: ['content.js']
        });
        // Wait a bit for the script to initialize
        await new Promise(resolve => setTimeout(resolve, 200));
      } catch (injectError) {
        throw injectError;
      }
    }
  }

  if (request.type === 'GET_PAGE_CONTEXT') {
    (async () => {
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs[0]?.id) {
          await ensureContentScript(tabs[0].id);
          const response = await chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_PAGE_CONTEXT' });
          sendResponse(response); // Return response directly, not wrapped
        } else {
          sendResponse({ success: false, error: 'No active tab found' });
        }
      } catch (error) {
        sendResponse({ success: false, error: (error as Error).message });
      }
    })();
    return true;
  }

  // Execute action on page
  if (request.type === 'EXECUTE_ACTION') {
    (async () => {
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs[0]?.id) {
          await ensureContentScript(tabs[0].id);
          const response = await chrome.tabs.sendMessage(tabs[0].id, {
            type: 'EXECUTE_ACTION',
            action: request.action,
            target: request.target,
            selector: request.selector,
            value: request.value,
            key: request.key,
            keys: request.keys,
            coordinates: request.coordinates,
            destination: request.destination,
            direction: request.direction,
            amount: request.amount
          });
          sendResponse(response);
        } else {
          sendResponse({ success: false, error: 'No active tab found' });
        }
      } catch (error) {
        sendResponse({ success: false, error: (error as Error).message });
      }
    })();
    return true;
  }

  // Take screenshot
  if (request.type === 'TAKE_SCREENSHOT') {
    (async () => {
      try {

        // Define restricted protocols (but allow regular web pages)
        const restrictedProtocols = ['chrome://', 'chrome-extension://', 'edge://', 'about:', 'devtools://'];
        const isRestricted = (url: string | undefined) => {
          if (!url) return true;
          // Allow http:// and https:// pages (including google.com)
          if (url.startsWith('http://') || url.startsWith('https://')) return false;
          // Block internal browser pages
          return restrictedProtocols.some(protocol => url.startsWith(protocol));
        };

        // Get the currently focused window with windowTypes to exclude devtools
        const currentWindow = await chrome.windows.getLastFocused({
          populate: true,
          windowTypes: ['normal']
        });

        if (!currentWindow || !currentWindow.tabs) {
          console.error('❌ No focused window found');
          sendResponse({ success: false, error: 'No browser window found' });
          return;
        }

        // Get the active AND highlighted tab (the one that's actually visible to the user)
        let activeTab = currentWindow.tabs.find(tab => tab.active === true && tab.highlighted === true);

        // Fallback to just active if highlighted not found
        if (!activeTab) {
          activeTab = currentWindow.tabs.find(tab => tab.active === true);
        }

        if (!activeTab) {
          console.error('❌ No active tab found in window');
          sendResponse({ success: false, error: 'No active tab found' });
          return;
        }


        // Check if the current tab is restricted
        if (isRestricted(activeTab.url)) {
          // Navigate to google.com automatically
          if (activeTab.id) {
            await chrome.tabs.update(activeTab.id, { url: 'https://www.google.com' });

            // Wait for the page to load
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Get the updated tab
            const updatedTab = await chrome.tabs.get(activeTab.id);

            // Update activeTab reference
            activeTab = updatedTab;
          } else {
            sendResponse({
              success: false,
              error: 'Cannot navigate from restricted page'
            });
            return;
          }
        }


        // Ensure windowId is defined
        if (currentWindow.id === undefined) {
          throw new Error('Window ID is undefined');
        }

        // Capture the visible tab in the current window
        const dataUrl = await chrome.tabs.captureVisibleTab(currentWindow.id, {
          format: 'png',
          quality: 80
        });

        sendResponse({ success: true, screenshot: dataUrl });
      } catch (error) {
        console.error('❌ Screenshot capture error:', error);

        // Provide more detailed error information
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error('Error details:', errorMsg);
        sendResponse({
          success: false,
          error: `Screenshot failed: ${errorMsg}`
        });
      }
    })().catch(err => {
      // Handle unhandled promise rejections from the IIFE
      console.error('[SCREENSHOT] Unhandled promise rejection:', err instanceof Error ? err.message : String(err));
    });
    return true;
  }

  // Navigate to URL
  if (request.type === 'NAVIGATE') {
    (async () => {
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs[0]?.id) {
          await chrome.tabs.update(tabs[0].id, { url: request.url });
          sendResponse({ success: true, url: request.url });
        } else {
          sendResponse({ success: false, error: 'No active tab found' });
        }
      } catch (error) {
        sendResponse({ success: false, error: (error as Error).message });
      }
    })();
    return true;
  }

  if (request.type === 'EXECUTE_SCRIPT') {
    sendResponse({
      success: false,
      error: 'EXECUTE_SCRIPT is disabled for security reasons. Use content script messaging instead.'
    });
    return true;
  }

  // Get bookmarks
  if (request.type === 'GET_BOOKMARKS') {
    chrome.bookmarks.getTree((bookmarkTree) => {
      sendResponse({ bookmarks: bookmarkTree });
    });
    return true;
  }

  // Store in memory
  if (request.type === 'STORE_MEMORY') {
    const { key, value } = request;
    memory.sessionData[key] = value;
    chrome.storage.local.set({ browserMemory: memory });
    sendResponse({ success: true });
    return true;
  }

  // Page loaded notification from content script
  if (request.type === 'PAGE_LOADED') {
    console.log('Page loaded:', request.url);
    return false;
  }
});

console.log('Theriad Browser background service worker loaded');
