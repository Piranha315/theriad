import type { Settings } from './types';
import type { TheriadBrowserContext } from './theriadClient';

const PAGE_TEXT_LIMIT = 12000;
const DOM_SUMMARY_LIMIT = 2500;

export async function buildBrowserContext(settings: Settings): Promise<TheriadBrowserContext | undefined> {
  if (!settings.includePageContext) {
    return undefined;
  }

  const tabInfo = await sendRuntimeMessage({ type: 'GET_TAB_INFO' }).catch(() => null);
  const context: TheriadBrowserContext = {
    url: tabInfo?.url,
    title: tabInfo?.title,
    screenshotAvailable: false,
  };

  if (isRestrictedUrl(context.url)) {
    context.pageText = 'Page context unavailable on restricted browser pages.';
    return context;
  }

  const pageContext = await sendRuntimeMessage({ type: 'GET_PAGE_CONTEXT' }).catch((error) => ({
    success: false,
    error: error instanceof Error ? error.message : String(error),
  }));

  if (pageContext && pageContext.success !== false) {
    context.url = pageContext.url || context.url;
    context.title = pageContext.title || context.title;
    context.pageText = limitText(pageContext.textContent || pageContext.text || '', PAGE_TEXT_LIMIT);
    context.selectionText = limitText(pageContext.selectionText || '', 4000) || undefined;
    context.domSummary = simplifyDomSummary(pageContext);
  } else if (pageContext?.error) {
    context.pageText = `Page context unavailable: ${pageContext.error}`;
  }

  if (settings.includeBrowserMemory) {
    const memoryResponse = await sendRuntimeMessage({ type: 'GET_MEMORY' }).catch(() => null);
    context.browserMemory = limitBrowserMemory(memoryResponse?.memory);
  }

  return context;
}

function sendRuntimeMessage(message: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      const error = chrome.runtime.lastError;
      if (error) {
        reject(new Error(error.message));
        return;
      }
      resolve(response);
    });
  });
}

function simplifyDomSummary(pageContext: any): unknown {
  return {
    links: (pageContext.links || []).slice(0, 25).map((link: any) => ({
      text: limitText(link.text || '', 120),
      href: link.href,
    })),
    images: (pageContext.images || []).slice(0, 10).map((image: any) => ({
      alt: limitText(image.alt || '', 120),
      src: image.src,
    })),
    forms: (pageContext.forms || []).slice(0, 8),
    metadata: pageContext.metadata || {},
    viewport: pageContext.viewport,
  };
}

function limitBrowserMemory(memory: any): unknown {
  if (!memory) return undefined;
  return {
    recentPages: (memory.recentPages || []).slice(0, 8).map((page: any) => ({
      url: page.url,
      title: page.title,
      timestamp: page.timestamp,
    })),
    userPreferences: memory.userPreferences || {},
    sessionData: JSON.stringify(memory.sessionData || {}).slice(0, DOM_SUMMARY_LIMIT),
  };
}

function limitText(text: string, limit: number): string {
  if (!text) return '';
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

function isRestrictedUrl(url?: string): boolean {
  if (!url) return true;
  return ['chrome://', 'chrome-extension://', 'edge://', 'about:', 'devtools://'].some((prefix) => url.startsWith(prefix));
}

