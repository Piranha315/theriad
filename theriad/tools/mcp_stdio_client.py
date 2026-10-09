"""
Minimal MCP JSON-RPC client over subprocess stdio.

Implements the subset of the Model Context Protocol needed for
ChromeDevToolsAdapter (Preview QA + browser-agent observation).

Wire format: one JSON-RPC object per newline, as required by MCP stdio.

Protocol sequence:
    1. Client → server: initialize request
    2. Server → client: initialize response
    3. Client → server: notifications/initialized  (no response expected)
    4. Client → server: tools/call requests
    5. Server → client: tools/call responses

Design notes
------------
- Fully synchronous; suitable for use inside a FastAPI route handler that
  wraps the call with asyncio.run_in_executor or calls it from a sync handler.
- One MCPStdioClient instance corresponds to one subprocess lifetime.
  Callers are responsible for calling close() when done (or using as a
  context manager).
- All reads are bounded by a per-message timeout; the process is killed if
  a timeout fires so we never block indefinitely.
- Stderr from the subprocess is captured and included in error messages to
  aid debugging.
- Never logs arguments or results — callers are responsible for audit logging.
"""
from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Protocol constants ────────────────────────────────────────────────────────

_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_NAME      = "theriad-mcp-client"
_CLIENT_VERSION   = "0.1.0"

# Default per-call timeout (seconds). Can be overridden per call.
DEFAULT_TIMEOUT = 30.0


class MCPError(Exception):
    """Raised when the MCP server returns a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code    = code
        self.message = message
        self.data    = data
        super().__init__(f"MCP error {code}: {message}")


class MCPStdioClient:
    """
    Synchronous MCP JSON-RPC client over subprocess stdio.

    Parameters
    ----------
    command:
        argv list for the subprocess (e.g. ["npx", "-y", "chrome-devtools-mcp@latest",
        "--isolated", "--headless"]).
    startup_timeout:
        Seconds to wait for the initialize handshake to complete.
    default_timeout:
        Default seconds to wait for each tools/call response.

    Usage
    -----
        with MCPStdioClient(["npx", "-y", "chrome-devtools-mcp@latest", "--isolated"]) as client:
            result = client.call_tool("navigate_page", {"url": "https://example.com"})
    """

    def __init__(
        self,
        command: list[str],
        startup_timeout: float = 15.0,
        default_timeout: float = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command         = command
        self._default_timeout = default_timeout
        self._next_id         = 1
        self._proc: subprocess.Popen | None = None
        self._initialized     = False

        # Merge env with os.environ; pass None to inherit fully.
        proc_env = {**os.environ, **(env or {})}

        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=proc_env,
            )
        except FileNotFoundError as exc:
            raise MCPError(-32000, f"MCP subprocess command not found: {command[0]}") from exc
        except OSError as exc:
            raise MCPError(-32000, f"Failed to start MCP subprocess: {exc}") from exc

        logger.debug("[mcp_stdio] started pid=%d cmd=%s", self._proc.pid, command[0])

        try:
            self._initialize(timeout=startup_timeout)
        except Exception:
            self._kill()
            raise

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MCPStdioClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Invoke a named MCP tool and return its result dict.

        Raises
        ------
        MCPError
            When the server returns a JSON-RPC error.
        TimeoutError
            When no response arrives within *timeout* seconds.
        RuntimeError
            When the subprocess has exited unexpectedly.
        """
        if not self._initialized:
            raise RuntimeError("MCPStdioClient is not initialized.")
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("MCP subprocess has exited.")

        req_id = self._next_id
        self._next_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }
        self._send(payload)

        response = self._recv(timeout=timeout or self._default_timeout)
        return self._unwrap(response, req_id)

    def close(self) -> None:
        """Terminate the subprocess and release resources."""
        self._kill()

    # ── Internal: MCP handshake ───────────────────────────────────────────────

    def _initialize(self, timeout: float) -> None:
        req_id = self._next_id
        self._next_id += 1

        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities":    {},
                "clientInfo": {
                    "name":    _CLIENT_NAME,
                    "version": _CLIENT_VERSION,
                },
            },
        })

        response = self._recv(timeout=timeout)
        self._unwrap(response, req_id)   # raises MCPError on JSON-RPC error

        # Notify the server that initialization is done (fire-and-forget)
        self._send({
            "jsonrpc": "2.0",
            "method":  "notifications/initialized",
        })

        self._initialized = True
        logger.debug("[mcp_stdio] initialized pid=%d", self._proc.pid)

    # ── Internal: wire I/O ────────────────────────────────────────────────────

    def _send(self, payload: dict[str, Any]) -> None:
        """Encode and write one JSON-RPC message to the subprocess stdin."""
        body = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(body)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("MCP subprocess stdin closed unexpectedly.") from exc

    def _recv(self, timeout: float) -> dict[str, Any]:
        """
        Read one newline-delimited JSON-RPC message from subprocess stdout.
        """
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MCP response timeout after {timeout}s (reading headers).")

            # select() lets us check readability without blocking forever
            rlist, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 1.0))
            if not rlist:
                # Check if subprocess died
                if self._proc.poll() is not None:
                    stderr = self._read_stderr()
                    raise RuntimeError(
                        f"MCP subprocess exited unexpectedly (rc={self._proc.returncode}). "
                        f"stderr: {stderr[:500]}"
                    )
                continue

            line = self._proc.stdout.readline()
            if not line:
                stderr = self._read_stderr()
                raise RuntimeError(
                    f"MCP subprocess stdout closed. stderr: {stderr[:500]}"
                )
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue
            try:
                message = json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON-RPC line from MCP server: {decoded[:200]}") from exc
            # Servers may emit notifications while a request is in flight.
            if "id" not in message:
                continue
            return message

    @staticmethod
    def _unwrap(response: dict[str, Any], expected_id: int | None) -> dict[str, Any]:
        """
        Extract the result from a JSON-RPC response, raising MCPError on errors.
        """
        if "error" in response:
            err = response["error"]
            raise MCPError(
                code=err.get("code", -1),
                message=err.get("message", "unknown error"),
                data=err.get("data"),
            )
        return response.get("result", {})

    def _read_stderr(self) -> str:
        """Non-blocking read of whatever is available on stderr."""
        try:
            rlist, _, _ = select.select([self._proc.stderr], [], [], 0.1)
            if rlist:
                return self._proc.stderr.read(4096).decode("utf-8", errors="replace")
        except Exception:
            pass
        return ""

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        except Exception:
            pass
        finally:
            self._proc = None
