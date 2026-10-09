"""Authenticated, task-bound bridge to the Theriad MV3 extension."""
from __future__ import annotations

import asyncio
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class ExtensionBridgeError(RuntimeError):
    pass


@dataclass
class Pairing:
    code: str
    task_id: str
    expires_at: float
    consumed: bool = False


@dataclass
class ExtensionConnection:
    task_id: str
    extension_id: str
    websocket: Any
    session_token: str
    assigned_tabs: set[int] = field(default_factory=set)
    sequence: int = 0
    last_seen: float = field(default_factory=time.time)
    pending: dict[str, asyncio.Future] = field(default_factory=dict, repr=False)
    last_result_sequence: int = 0


class ExtensionBridge:
    def __init__(self) -> None:
        self._pairings: dict[str, Pairing] = {}
        self._connections: dict[str, ExtensionConnection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()

    def create_pairing(self, task_id: str, *, ttl_seconds: int = 300) -> dict:
        code = "-".join((secrets.token_hex(2), secrets.token_hex(2))).upper()
        pairing = Pairing(code, task_id, time.time() + max(30, min(ttl_seconds, 900)))
        with self._lock:
            self._pairings[code] = pairing
        return {
            "task_id": task_id,
            "pairing_code": code,
            "expires_at": pairing.expires_at,
            "instructions": "Enter this one-time code in the Theriad extension and choose Pair current tab.",
        }

    @staticmethod
    def extension_id_from_origin(origin: str | None) -> str:
        prefix = "chrome-extension://"
        if not origin or not origin.startswith(prefix):
            raise ExtensionBridgeError("Only chrome-extension origins may connect.")
        extension_id = origin[len(prefix):].strip("/")
        if not extension_id or "/" in extension_id:
            raise ExtensionBridgeError("Invalid Chrome extension origin.")
        configured = os.environ.get("THERIAD_EXTENSION_ID", "").strip()
        if configured and extension_id != configured:
            raise ExtensionBridgeError("Chrome extension ID is not allowed.")
        return extension_id

    async def accept(self, websocket: Any, code: str) -> None:
        extension_id = self.extension_id_from_origin(websocket.headers.get("origin"))
        with self._lock:
            pairing = self._pairings.get(code.upper())
            if pairing is None or pairing.consumed or pairing.expires_at < time.time():
                raise ExtensionBridgeError("Pairing code is invalid, expired, or already used.")
            pairing.consumed = True
        await websocket.accept()
        self._loop = asyncio.get_running_loop()
        connection = ExtensionConnection(
            task_id=pairing.task_id,
            extension_id=extension_id,
            websocket=websocket,
            session_token=secrets.token_urlsafe(32),
        )
        self._connections[pairing.task_id] = connection
        await websocket.send_json({
            "type": "paired",
            "task_id": pairing.task_id,
            "session_token": connection.session_token,
        })
        try:
            while True:
                message = await websocket.receive_json()
                await self._handle(connection, message)
        finally:
            if self._connections.get(pairing.task_id) is connection:
                self._connections.pop(pairing.task_id, None)
            for future in connection.pending.values():
                if not future.done():
                    future.set_exception(ExtensionBridgeError("Extension disconnected."))

    async def _handle(self, connection: ExtensionConnection, message: dict[str, Any]) -> None:
        if len(str(message)) > 1_000_000:
            raise ExtensionBridgeError("Extension message exceeds size limit.")
        connection.last_seen = time.time()
        kind = message.get("type")
        if kind == "hello":
            tab_id = message.get("tab_id")
            if not isinstance(tab_id, int):
                raise ExtensionBridgeError("hello requires the explicitly assigned tab_id.")
            connection.assigned_tabs = {tab_id}
            await connection.websocket.send_json({
                "type": "ready", "task_id": connection.task_id, "assigned_tabs": [tab_id]
            })
            return
        if message.get("session_token") != connection.session_token:
            raise ExtensionBridgeError("Invalid extension session token.")
        if kind == "heartbeat":
            await connection.websocket.send_json({"type": "heartbeat_ack", "at": time.time()})
            return
        if kind == "result":
            sequence = int(message.get("sequence") or 0)
            if sequence <= connection.last_result_sequence:
                raise ExtensionBridgeError("Rejected replayed or out-of-order extension result.")
            connection.last_result_sequence = sequence
            request_id = str(message.get("request_id") or "")
            future = connection.pending.pop(request_id, None)
            if future is not None and not future.done():
                future.set_result(message)

    async def call(self, task_id: str, tool: str, arguments: dict[str, Any], *, timeout: float = 30.0) -> dict:
        connection = self._connections.get(task_id)
        if connection is None or not connection.assigned_tabs:
            raise ExtensionBridgeError("No paired extension tab for this task.")
        tab_id = int(arguments.get("tab_id") or next(iter(connection.assigned_tabs)))
        if tab_id not in connection.assigned_tabs:
            raise ExtensionBridgeError("Requested tab is not assigned to this task.")
        connection.sequence += 1
        request_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        connection.pending[request_id] = future
        await connection.websocket.send_json({
            "type": "command",
            "request_id": request_id,
            "sequence": connection.sequence,
            "session_token": connection.session_token,
            "task_id": task_id,
            "tab_id": tab_id,
            "tool": tool,
            "arguments": arguments,
        })
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        finally:
            connection.pending.pop(request_id, None)
        if not result.get("ok"):
            raise ExtensionBridgeError(str(result.get("error") or "Extension tool failed."))
        output = result.get("result") or {}
        if tool == "new_tab" and isinstance(output.get("tab_id"), int):
            connection.assigned_tabs.add(output["tab_id"])
        elif tool == "close_tab":
            connection.assigned_tabs.discard(tab_id)
        return output

    def call_sync(self, task_id: str, tool: str, arguments: dict[str, Any]) -> dict:
        if self._loop is None or not self._loop.is_running():
            raise ExtensionBridgeError("Extension bridge event loop is unavailable.")
        future = asyncio.run_coroutine_threadsafe(self.call(task_id, tool, arguments), self._loop)
        return future.result(timeout=35)

    def status(self, task_id: str | None = None) -> dict:
        connections = [
            {
                "task_id": item.task_id,
                "extension_id": item.extension_id,
                "assigned_tabs": sorted(item.assigned_tabs),
                "last_seen": item.last_seen,
            }
            for item in self._connections.values()
            if task_id is None or item.task_id == task_id
        ]
        return {
            "paired": bool(connections),
            "available": bool(connections),
            "connections": connections,
            "allowed_extension_id_configured": bool(os.environ.get("THERIAD_EXTENSION_ID")),
        }


_BRIDGE = ExtensionBridge()


def get_extension_bridge() -> ExtensionBridge:
    return _BRIDGE
