const NETWORK_ERROR_RE = /failed to fetch|networkerror|load failed|econnrefused|fetch failed|can't reach/i;
const API_ERROR_RE = /^API\s+\d{3}\b/i;

export function toFriendlyError(error, fallback = 'That did not work. Please try again.') {
  const message = typeof error === 'string'
    ? error
    : error?.message || '';
  const status = Number(error?.status || message.match(/^API\s+(\d{3})/)?.[1] || 0);

  if (NETWORK_ERROR_RE.test(message)) {
    return 'Theriad is having trouble reaching the workspace. Check your connection and try again.';
  }
  if (status === 401 || /unauthorized|session expired/i.test(message)) {
    return 'Your session needs attention. Refresh the page and sign in again.';
  }
  if (status === 403 || /forbidden/i.test(message)) {
    return 'This action needs more access. Create an account or check your permissions, then try again.';
  }
  if (status === 402 && typeof error?.serverMessage === 'string' && error.serverMessage.trim()) {
    // Allowance/limit messages are written for the user; show them as-is.
    return error.serverMessage.trim();
  }
  if (status >= 500 || API_ERROR_RE.test(message)) {
    // Surface the server's own explanation when it gave a short, plain one.
    const detail = typeof error?.serverMessage === 'string' ? error.serverMessage.trim() : '';
    if (detail && detail.length <= 240 && !/traceback|\{\s*"/i.test(detail)) {
      return `Theriad could not complete that request: ${detail}`;
    }
    return 'Theriad could not complete that request. Please try again in a moment.';
  }

  return message || fallback;
}
