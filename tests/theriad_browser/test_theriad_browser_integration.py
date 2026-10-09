from pathlib import Path

from fastapi.testclient import TestClient

from theriad.api.server import app


ROOT = Path(__file__).resolve().parents[1]
THERIAD_BROWSER = ROOT / "apps" / "theriad-extension"


def test_extension_chat_send_returns_router_metadata():
    client = TestClient(app)

    response = client.post(
        "/api/chat/send",
        json={
            "messages": [{"role": "user", "content": "Summarize this page."}],
            "mode": "chat",
            "browserContext": {
                "url": "https://example.com",
                "title": "Example Domain",
                "pageText": "Example Domain is a page used for examples.",
            },
            "runtimeProfile": "consumer_secure",
            "client": {"name": "theriad-extension", "version": "1.0.0"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"]
    assert body["model"]
    assert body["provider"] == "openrouter"
    assert body["routeTrace"]["taskType"] == "page_summary"
    assert body["cost"]["inputTokens"] > 0


def test_extension_chat_send_warns_for_browser_actions_without_tools():
    client = TestClient(app)

    response = client.post(
        "/api/chat/send",
        json={
            "messages": [{"role": "user", "content": "Click the checkout button and submit payment."}],
            "mode": "chat",
            "runtimeProfile": "consumer_secure",
            "requestedCapabilities": [],
            "client": {"name": "theriad-extension"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["routeTrace"]["taskType"] == "sensitive_action"
    assert "requires Browser Tools mode" in body["warnings"][0]


def test_health_and_status_endpoints_support_extension_detection():
    client = TestClient(app)

    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/tools/status").json()["ok"] is True
    assert client.get("/api/provider/status").json()["ok"] is True


def test_theriad_extension_contains_theriad_browser_adapter_and_feature_flags():
    sidepanel = (THERIAD_BROWSER / "sidepanel.tsx").read_text(encoding="utf-8")
    settings = (THERIAD_BROWSER / "settings.tsx").read_text(encoding="utf-8")
    client = (THERIAD_BROWSER / "theriadClient.ts").read_text(encoding="utf-8")
    browser_context = (THERIAD_BROWSER / "browserContext.ts").read_text(encoding="utf-8")
    manifest = (THERIAD_BROWSER / "manifest.json").read_text(encoding="utf-8")

    assert "sendTheriadRequest" in sidepanel
    assert "buildBrowserContext" in sidepanel
    assert "Browser Tools active" in sidepanel
    assert "enableComposioTools" in sidepanel
    assert "Test Theriad Connection" in settings
    assert "Google API Key" in settings
    assert "Composio API Key" in settings
    assert "/api/chat/send" in client
    assert "/api/execute" in client
    assert "PAGE_TEXT_LIMIT = 12000" in browser_context
    assert "Theriad Browser" in manifest

