"""Opt-in macOS window-scoped computer-use helper.

The adapter is unavailable unless explicitly enabled and PyObjC Quartz/AppKit
are installed. It refuses input unless the assigned application is frontmost.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from typing import Any


class DesktopControlError(RuntimeError):
    pass


def _enabled() -> bool:
    return os.environ.get("THERIAD_DESKTOP_CONTROL_ENABLED", "").lower() in {"1", "true", "yes"}


def availability() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"supported": False, "enabled": _enabled(), "available": False, "error": "macOS is required."}
    try:
        import AppKit  # noqa: F401
        import Quartz  # noqa: F401
    except ImportError:
        return {
            "supported": True,
            "enabled": _enabled(),
            "available": False,
            "error": "Install the desktop extra and grant Accessibility plus Screen Recording permissions.",
        }
    return {"supported": True, "enabled": _enabled(), "available": _enabled(), "error": None if _enabled() else "Desktop control is disabled."}


@dataclass
class DesktopSession:
    task_id: str
    application: str
    stopped: bool = False


class MacOSComputerUse:
    def __init__(self) -> None:
        self._sessions: dict[str, DesktopSession] = {}
        self._lock = threading.RLock()

    def assign(self, task_id: str, application: str, allowed: list[str]) -> dict:
        state = availability()
        if not state["available"]:
            raise DesktopControlError(state["error"] or "Desktop control unavailable.")
        if application not in allowed:
            raise DesktopControlError("Application is not in this task's explicit allowlist.")
        subprocess.run(
            ["osascript", "-e", f"tell application {json.dumps(application)} to activate"],
            check=True,
            timeout=5,
        )
        self._sessions[task_id] = DesktopSession(task_id, application)
        return {"task_id": task_id, "application": application, "active_control": True}

    def _session(self, task_id: str) -> DesktopSession:
        session = self._sessions.get(task_id)
        if session is None or session.stopped:
            raise DesktopControlError("No active desktop assignment for this task.")
        from AppKit import NSWorkspace
        active = NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
        if active != session.application:
            raise DesktopControlError(
                f"Assigned application is not frontmost (expected {session.application}, found {active})."
            )
        return session

    def screenshot(self, task_id: str) -> dict:
        session = self._session(task_id)
        import AppKit
        import Quartz
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        match = next(
            (item for item in windows if item.get(Quartz.kCGWindowOwnerName) == session.application and item.get(Quartz.kCGWindowLayer, 1) == 0),
            None,
        )
        if match is None:
            raise DesktopControlError("No visible window found for the assigned application.")
        window_id = int(match[Quartz.kCGWindowNumber])
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )
        if image is None:
            raise DesktopControlError("Screen Recording permission is required.")
        bitmap = AppKit.NSBitmapImageRep.alloc().initWithCGImage_(image)
        data = bitmap.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
        return {
            "application": session.application,
            "window_id": window_id,
            "png_base64": base64.b64encode(bytes(data)).decode("ascii"),
        }

    def click(self, task_id: str, x: float, y: float) -> dict:
        session = self._session(task_id)
        import Quartz
        point = Quartz.CGPoint(float(x), float(y))
        for event_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            event = Quartz.CGEventCreateMouseEvent(None, event_type, point, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return {"application": session.application, "clicked": {"x": x, "y": y}}

    def type_text(self, task_id: str, text: str) -> dict:
        session = self._session(task_id)
        import Quartz
        for pressed in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, 0, pressed)
            Quartz.CGEventKeyboardSetUnicodeString(event, len(text), text)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return {"application": session.application, "characters_typed": len(text)}

    def stop(self, task_id: str) -> dict:
        session = self._sessions.get(task_id)
        if session:
            session.stopped = True
        return {"task_id": task_id, "active_control": False}


_COMPUTER = MacOSComputerUse()


def get_computer_use() -> MacOSComputerUse:
    return _COMPUTER
