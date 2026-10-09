from __future__ import annotations

import pytest

from theriad.browser.extension_bridge import ExtensionBridge, ExtensionBridgeError


def test_only_chrome_extension_origins_are_accepted(monkeypatch):
    bridge = ExtensionBridge()
    with pytest.raises(ExtensionBridgeError):
        bridge.extension_id_from_origin("https://evil.example")
    assert bridge.extension_id_from_origin("chrome-extension://abcdefghijklmnop") == "abcdefghijklmnop"


def test_configured_extension_id_is_enforced(monkeypatch):
    bridge = ExtensionBridge()
    monkeypatch.setenv("THERIAD_EXTENSION_ID", "allowed")
    with pytest.raises(ExtensionBridgeError):
        bridge.extension_id_from_origin("chrome-extension://different")
    assert bridge.extension_id_from_origin("chrome-extension://allowed") == "allowed"


def test_pairings_expire_and_are_task_bound():
    bridge = ExtensionBridge()
    pairing = bridge.create_pairing("task-1", ttl_seconds=60)
    internal = bridge._pairings[pairing["pairing_code"]]
    assert internal.task_id == "task-1"
    assert internal.consumed is False
