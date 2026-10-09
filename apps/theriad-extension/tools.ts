
/**
 * Generates or retrieves a unique, persistent user ID for this extension installation
 * This prevents all users from sharing the same session/rate limits
 *
 * Uses UUID v4 for uniqueness and stores in chrome.storage for persistence
 */
export async function getOrCreateUserId(): Promise<string> {
  try {
    const stored = await chrome.storage.local.get('extensionUserId');

    if (stored.extensionUserId) {
      return stored.extensionUserId;
    }

    // Generate a new unique ID using crypto.randomUUID (available in modern browsers)
    const newUserId = `theriad-${crypto.randomUUID()}`;

    // Store for future use
    await chrome.storage.local.set({ extensionUserId: newUserId });

    console.log('Generated new unique extension user ID:', newUserId);
    return newUserId;
  } catch (error) {
    console.error('Error generating user ID, falling back to timestamp-based ID:', error);
    // Fallback if crypto is not available (shouldn't happen in modern browsers)
    return `theriad-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  }
}

export async function initializeComposioToolRouter(apiKey: string, useUserId?: string) {
  try {
    const userId = useUserId || await getOrCreateUserId();

    const response = await fetch('https://backend.composio.dev/api/v3/tool_router/session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
      },
      body: JSON.stringify({ user_id: userId }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to create Composio session: ${response.status} ${errorText}`);
    }

    const session = await response.json();
    const sessionExpiresAt = Date.now() + (24 * 60 * 60 * 1000); // 24 hour expiration

    return {
      sessionId: session.session_id,
      mcpUrl: session.mcp.url,
      expiresAt: sessionExpiresAt,
      createdAt: Date.now(),
    };
  } catch (error) {
    console.error('Error initializing Composio Tool Router:', error);
    throw error;
  }
}