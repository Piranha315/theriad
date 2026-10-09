"""Compatibility import for the canonical browser-session adapter.

The former chat-only runtime has been retired.  Importing this module now
returns the policy-gated session types used by ``/api/browser/session``.
"""

from .browser_agent import BrowserAgentSession, BrowserAgentStore


class BrowserAgentRuntime(BrowserAgentStore):
    """Deprecated alias; new callers should use the session API."""
