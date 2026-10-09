"""Legacy development fixture server.

The supported product is the React app plus ``theriad.api.server``.  This
server remains only for fixture/manual compatibility checks and intentionally
does not expose a competing root UI.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .browser_agent import BrowserAgentStartRequest, STORE
from .chrome import chrome_status
from .runtime_config import current_config

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: object, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/tools/chrome/status":
            self._send(200, chrome_status())
            return
        if parsed.path == "/api/settings/runtime":
            self._send(200, current_config().safe_status())
            return
        if parsed.path == "/api/execute":
            self._send(200, {"ok": True, "message": "execute route unaffected by runtime profile"})
            return
        if parsed.path == "/preview":
            self._send(200, b"<iframe sandbox=\"allow-scripts\" srcdoc=\"Preview is isolated from browser-agent session data.\"></iframe>", "text/html")
            return
        if parsed.path == "/api/agent/browser/status":
            session_id = parse_qs(parsed.query).get("session_id", [""])[0]
            try:
                self._send(200, STORE.get(session_id).to_dict())
            except KeyError:
                self._send(404, {"error": "unknown session_id"})
            return
        path = STATIC_DIR / "index.html"
        if parsed.path == "/dev-fixtures/browser-agent" and path.is_file():
            self._send(200, path.read_bytes(), "text/html" if path.suffix == ".html" else "text/plain")
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        # Deprecated aliases retained for existing local scripts.  New clients
        # use /api/browser/session on theriad.api.server.
        if self.path == "/api/agent/browser/start":
            session = STORE.start(BrowserAgentStartRequest(**self._json()))
            self._send(200, session.to_dict())
            return
        payload = self._json()
        session_id = payload.get("session_id")
        try:
            session = STORE.get(session_id)
        except KeyError:
            self._send(404, {"error": "unknown session_id"})
            return
        if self.path == "/api/agent/browser/approve":
            session.approve()
        elif self.path == "/api/agent/browser/deny":
            session.deny()
        elif self.path == "/api/agent/browser/stop":
            session.stop()
        elif self.path == "/api/agent/browser/use-api-search":
            session.use_api_search()
        elif self.path == "/api/agent/browser/debug-step":
            session.tick()
        else:
            self._send(404, {"error": "not found"})
            return
        self._send(200, session.to_dict())


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    run()
