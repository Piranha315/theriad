"""
Built-in MCP server for OpenAI-compatible LLM providers.

Run as a subprocess to expose any provider that supports the OpenAI Chat
Completions API as an MCP tool — Groq, DeepInfra, Together AI, local
Ollama, etc.  Launched by LLMMCPAdapter; not intended to be run by hand.

Protocol: MCP JSON-RPC over stdio (newline-delimited).

Single tool exposed
-------------------
    complete — send a prompt to the configured provider and return the
               model's text response.

Configuration (env vars, set by LLMMCPAdapter)
----------------------------------------------
    LLM_MCP_BASE_URL   Required.  OpenAI-compatible base URL.
                       e.g. "https://api.groq.com/openai/v1"
    LLM_MCP_API_KEY    Required.  Bearer token.
    LLM_MCP_MODEL      Required.  Model identifier string.
    LLM_MCP_PROVIDER   Optional.  Human-readable name used in tool description.
    LLM_MCP_TIMEOUT    Optional.  Per-request timeout in seconds (default 30).
"""
from __future__ import annotations

import json
import os
import sys


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


BASE_URL = _env("LLM_MCP_BASE_URL")
API_KEY  = _env("LLM_MCP_API_KEY")
MODEL    = _env("LLM_MCP_MODEL")
PROVIDER = _env("LLM_MCP_PROVIDER", "LLM")
TIMEOUT  = float(_env("LLM_MCP_TIMEOUT", "30"))

_TOOL_NAME = "complete"
_TOOL_DESC = (
    f"Send a prompt to {PROVIDER} ({MODEL}) and return the model's text response. "
    "Use this to get a second opinion, perform a subtask with a different model, "
    "or call a cost-efficient model for a simpler step."
)

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The user prompt to send to the model.",
        },
        "system": {
            "type": "string",
            "description": "Optional system message.",
        },
        "model": {
            "type": "string",
            "description": f"Override the model. Defaults to {MODEL!r}.",
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum tokens to generate (default 1024).",
        },
    },
    "required": ["prompt"],
}


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _complete(args: dict) -> str:
    prompt   = args.get("prompt", "")
    system   = args.get("system", "")
    model    = args.get("model") or MODEL
    max_tok  = int(args.get("max_tokens", 1024))

    if not BASE_URL or not API_KEY or not model:
        raise RuntimeError(
            "LLM_MCP_BASE_URL, LLM_MCP_API_KEY, and LLM_MCP_MODEL must be set."
        )

    import urllib.request

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tok,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    import socket
    ctx = None
    try:
        import ssl
        ctx = ssl.create_default_context()
    except Exception:
        pass

    with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
        body = json.loads(resp.read())

    return body["choices"][0]["message"]["content"]


def main() -> None:
    _id = 1

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        req_id = msg.get("id")

        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": f"{PROVIDER}-mcp", "version": "1.0.0"},
                },
            })

        elif method == "notifications/initialized":
            pass  # fire-and-forget, no response needed

        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [{
                        "name": _TOOL_NAME,
                        "description": _TOOL_DESC,
                        "inputSchema": _TOOL_SCHEMA,
                    }]
                },
            })

        elif method == "tools/call":
            params   = msg.get("params", {})
            name     = params.get("name")
            args     = params.get("arguments", {})
            if name != _TOOL_NAME:
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"},
                })
                continue
            try:
                text = _complete(args)
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                })
            except Exception as exc:
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                })

        else:
            if req_id is not None:
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })


if __name__ == "__main__":
    main()
