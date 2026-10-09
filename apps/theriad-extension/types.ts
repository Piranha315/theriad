// Shared types for the extension
import { z } from 'zod';

export type ToolMode = 'tool-router';
export type TheriadMode = 'router' | 'execute' | 'chat';
export type RuntimeProfile = 'consumer_secure' | 'personal_trusted';

export interface Settings {
  provider: 'theriad' | 'google';
  apiKey: string;
  model: string;
  toolMode?: ToolMode;
  composioApiKey?: string;
  theriadApiUrl: string;
  theriadMode: TheriadMode;
  includePageContext: boolean;
  includeBrowserMemory: boolean;
  showRouteTrace: boolean;
  runtimeProfile: RuntimeProfile;
  enableComposioTools?: boolean;
  agentBridgeEnabled?: boolean;
  agentBridgePairingCode?: string;
}

export const DEFAULT_SETTINGS: Settings = {
  provider: 'theriad',
  apiKey: '',
  model: 'theriad-router',
  toolMode: 'tool-router',
  composioApiKey: '',
  theriadApiUrl: 'http://localhost:8000',
  theriadMode: 'chat',
  includePageContext: true,
  includeBrowserMemory: false,
  showRouteTrace: true,
  runtimeProfile: 'consumer_secure',
  enableComposioTools: false,
  agentBridgeEnabled: false,
  agentBridgePairingCode: '',
};

export interface ComposioSession {
  sessionId: string;
  mcpUrl: string;
  expiresAt: number;
  createdAt: number;
}

export interface ChatState {
  phase: 'loading' | 'ready' | 'streaming' | 'error';
  settings: Settings | null;
  messages: Message[];
  error: string | null;
  isLoading: boolean;
  browserToolsEnabled: boolean;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: GeminiFunctionCall[];
  model?: string;
  provider?: string;
  routeTrace?: unknown;
  cost?: {
    inputTokens?: number;
    outputTokens?: number;
    estimatedUsd?: number;
  };
  warnings?: string[];
}

export interface PageContext {
  url: string;
  title: string;
  textContent: string;
  selectionText?: string;
  links: Array<{ text: string; href: string }>;
  images: Array<{ alt: string; src: string }>;
  forms: Array<{
    id: string;
    action: string;
    inputs: Array<{ name: string; type: string }>
  }>;
  metadata: {
    description?: string;
    keywords?: string;
    author?: string;
  };
  viewport?: {
    width: number;
    height: number;
    scrollX: number;
    scrollY: number;
    devicePixelRatio: number;
  };
}

export interface BrowserMemory {
  recentPages: Array<{
    url: string;
    title: string;
    timestamp: number;
    context?: any
  }>;
  userPreferences: Record<string, any>;
  sessionData: Record<string, any>;
}

export interface MessageRequest {
  type: string;
  [key: string]: any;
}

export interface MessageResponse {
  success?: boolean;
  error?: string;
  [key: string]: any;
}

export interface ToolCall {
  id: string;
  name: string;
  parameters: Record<string, unknown>;
}

export interface ToolResult {
  toolCallId: string;
  result: unknown;
}

/**
 * MCP Client type for managing Model Context Protocol connections
 * Matches the AI SDK createMCPClient return type
 */
export interface MCPClient {
  tools(): Promise<Record<string, any>>;
  close(): Promise<void>;
}

/**
 * Browser action function parameters
 */
export interface BrowserActionParams {
  x?: number;
  y?: number;
  text?: string;
  selector?: string;
  target?: string;
  value?: string;
  direction?: string;
  amount?: number;
  key?: string;
  keys?: string[];
  destination_x?: number;
  destination_y?: number;
  coordinate?: { x: number; y: number };
  address?: string;
  uri?: string;
  content?: string;
  seconds?: number;
  milliseconds?: number;
  press_enter?: boolean;
  clear_before_typing?: boolean;
  magnitude?: number;
}

/**
 * Gemini API function call
 */
export interface GeminiFunctionCall {
  name: string;
  args?: Record<string, unknown>;
}

/**
 * Extended viewport info
 */
export interface ViewportInfo {
  width: number;
  height: number;
  scrollX: number;
  scrollY: number;
  devicePixelRatio: number;
}

// ============================================
// Zod Validation Schemas (for runtime validation)
// ============================================

/**
 * Validates Gemini API response structure
 * Ensures response has expected format before processing
 */
export const GeminiPartSchema = z.union([
  z.object({
    text: z.string(),
  }),
  z.object({
    functionCall: z.object({
      name: z.string(),
      args: z.record(z.any()).optional(),
    }),
  }),
  z.object({
    function_response: z.any(),
  }),
  z.object({
    inline_data: z.object({
      mime_type: z.string(),
      data: z.string(),
    }),
  }),
]);

export const GeminiCandidateSchema = z.object({
  content: z.object({
    parts: z.array(GeminiPartSchema),
  }).optional(),
  finishReason: z.string().optional(),
  safetyResponse: z.object({
    requireConfirmation: z.boolean(),
    message: z.string().optional(),
  }).optional(),
});

export const GeminiResponseSchema = z.object({
  candidates: z.array(GeminiCandidateSchema).optional(),
  promptFeedback: z.object({
    blockReason: z.string().optional(),
  }).optional(),
});

/**
 * Validates Composio session response
 */
export const ComposioSessionSchema = z.object({
  session_id: z.string().min(1),
  mcp: z.object({
    url: z.string().url(),
  }),
});

/**
 * Validates page context from content script
 */
export const PageContextSchema = z.object({
  url: z.string().url().or(z.string().startsWith('data:')),
  title: z.string(),
  textContent: z.string(),
  selectionText: z.string().optional(),
  links: z.array(z.object({
    text: z.string(),
    href: z.string(),
  })).default([]),
  images: z.array(z.object({
    alt: z.string(),
    src: z.string(),
  })).default([]),
  forms: z.array(z.object({
    id: z.string(),
    action: z.string(),
    inputs: z.array(z.object({
      name: z.string(),
      type: z.string(),
    })),
  })).default([]),
  metadata: z.object({
    description: z.string().optional(),
    keywords: z.string().optional(),
    author: z.string().optional(),
  }).optional(),
  viewport: z.object({
    width: z.number().positive(),
    height: z.number().positive(),
    scrollX: z.number(),
    scrollY: z.number(),
    devicePixelRatio: z.number().positive(),
  }).optional(),
});

/**
 * Validates action response from content script
 */
export const ActionResponseSchema = z.object({
  success: z.boolean(),
  message: z.string().optional(),
  error: z.string().optional(),
  data: z.any().optional(),
  element: z.string().optional(),
  elementBounds: z.object({
    left: z.number(),
    top: z.number(),
    width: z.number(),
    height: z.number(),
  }).optional(),
  text: z.string().optional(),
  screenshot: z.string().optional(),
});

/**
 * Validates screenshot response
 */
export const ScreenshotResponseSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),
  screenshot: z.string().optional(), // data URL
});
