import { z } from 'zod';

export interface ToolCall {
  id: string;
  toolName: string;
  args?: Record<string, unknown>;
  result?: unknown;
  status: 'pending' | 'completed' | 'failed';
}

export interface Attachment {
  name: string;
  type: string;       // MIME type
  dataUrl: string;    // base64 data URL (for images/PDFs) or '' for text-only
  size: number;
  textContent?: string; // populated for text files
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  mode: 'chat' | 'web';
  timestamp?: number;
  toolCalls?: ToolCall[];
  attachments?: Attachment[];
}

export interface Settings {
  googleApiKey: string;
  composioApiKey?: string;
  model: string;
}

export interface BrowserAction {
  type: 'navigate' | 'click' | 'type' | 'scroll' | 'screenshot';
  url?: string;
  x?: number;
  y?: number;
  text?: string;
  direction?: 'up' | 'down';
  amount?: number;
}

// SECURITY: Zod schemas for validating data from localStorage
// This prevents corrupted or malicious data from being used

export const ToolCallSchema = z.object({
  id: z.string(),
  toolName: z.string(),
  args: z.record(z.unknown()).optional(),
  result: z.unknown().optional(),
  status: z.enum(['pending', 'completed', 'failed']),
});

export const AttachmentSchema = z.object({
  name: z.string(),
  type: z.string(),
  dataUrl: z.string(),
  size: z.number(),
  textContent: z.string().optional(),
});

export const MessageSchema = z.object({
  id: z.string(),
  role: z.enum(['user', 'assistant']),
  content: z.string(),
  mode: z.enum(['chat', 'web']),
  timestamp: z.number().optional(),
  toolCalls: z.array(ToolCallSchema).optional(),
  attachments: z.array(AttachmentSchema).optional(),
});

export const ChatHistorySchema = z.array(MessageSchema);

export const SettingsSchema = z.object({
  googleApiKey: z.string(),
  composioApiKey: z.string().optional(),
  model: z.string(),
});
