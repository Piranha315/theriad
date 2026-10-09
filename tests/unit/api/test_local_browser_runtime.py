"""Persistent Playwright Chromium runtime API tests."""
from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from theriad.api.server import app


TEST_PAGE = """<!doctype html>
<html>
  <head><title>Theriad browser test</title></head>
  <body>
    <main>
      <h1>Runtime fixture</h1>
      <label for="name">Name</label>
      <input id="name" placeholder="Your name">
      <label for="password">Password</label>
      <input id="password" type="password">
      <button id="safe" onclick="document.querySelector('h1').textContent='Clicked'">Click me</button>
      <button id="send" type="submit">Send message</button>
    </main>
  </body>
</html>"""


@pytest.fixture()
def client():
    with TestClient(app) as value:
        yield value


@pytest.fixture()
def browser_session(client):
    response = client.post("/api/browser/session/start", json={"visible": False})
    data = response.json()
    if not data.get("ok"):
        pytest.skip(data.get("error") or "Playwright Chromium unavailable")
    session_id = data["session_id"]
    try:
        yield session_id
    finally:
        client.post("/api/browser/session/stop", json={"session_id": session_id})


def _navigate_fixture(client, session_id):
    url = "data:text/html," + quote(TEST_PAGE)
    response = client.post(
        "/api/browser/navigate",
        json={"session_id": session_id, "url": url},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    return data


def _element(data, *, text=None, field_type=None):
    items = data["clickable_elements"] if text else data["input_fields"]
    for item in items:
        if text and item.get("text") == text:
            return item
        if field_type and item.get("type") == field_type:
            return item
    raise AssertionError(f"Element not found: text={text!r}, field_type={field_type!r}")


def test_session_start_status_and_stop(client):
    response = client.post("/api/browser/session/start", json={"visible": False})
    data = response.json()
    if not data.get("ok"):
        pytest.skip(data.get("error") or "Playwright Chromium unavailable")
    session_id = data["session_id"]
    assert data["browser"] == "chromium"
    assert data["backend"] == "opensandbox"

    status = client.get("/api/browser/status", params={"session_id": session_id}).json()
    assert status["status"] == "running"
    assert status["browser_backend"] == "opensandbox"

    stopped = client.post(
        "/api/browser/session/stop", json={"session_id": session_id}
    ).json()
    assert stopped["status"] == "stopped"


def test_navigate_and_extract_stable_elements(client, browser_session):
    first = _navigate_fixture(client, browser_session)
    assert first["page_title"] == "Theriad browser test"
    assert first["current_url"].startswith("data:text/html")
    assert first["visible_text_summary"]
    assert first["accessibility_tree_summary"]
    assert first["screenshot_base64"]

    button = _element(first, text="Click me")
    name = _element(first, field_type="text")
    second = client.get(
        "/api/browser/observe",
        params={"session_id": browser_session, "screenshot": False},
    ).json()
    assert _element(second, text="Click me")["id"] == button["id"]
    assert _element(second, field_type="text")["id"] == name["id"]


def test_click_button_and_fill_form(client, browser_session):
    observation = _navigate_fixture(client, browser_session)
    button_id = _element(observation, text="Click me")["id"]
    name_id = _element(observation, field_type="text")["id"]

    clicked = client.post("/api/browser/action", json={
        "session_id": browser_session,
        "action": "click",
        "element_id": button_id,
    }).json()
    assert clicked["ok"] is True
    assert "Clicked" in clicked["visible_text_summary"]

    filled = client.post("/api/browser/action", json={
        "session_id": browser_session,
        "action": "type",
        "element_id": name_id,
        "text": "Theriad",
    }).json()
    assert filled["ok"] is True


def test_policy_blocks_sensitive_field_and_secret_extraction(client, browser_session):
    observation = _navigate_fixture(client, browser_session)
    password_id = _element(observation, field_type="password")["id"]

    blocked_fill = client.post("/api/browser/action", json={
        "session_id": browser_session,
        "action": "type",
        "element_id": password_id,
        "text": "do-not-enter",
    }).json()
    assert blocked_fill["status"] == "blocked"

    blocked_extract = client.post("/api/browser/action", json={
        "session_id": browser_session,
        "action": "extract_api_key",
    }).json()
    assert blocked_extract["status"] == "blocked"


def test_approval_required_action_pauses_then_can_be_denied(client, browser_session):
    observation = _navigate_fixture(client, browser_session)
    send_id = _element(observation, text="Send message")["id"]

    pending = client.post("/api/browser/action", json={
        "session_id": browser_session,
        "action": "click",
        "element_id": send_id,
    }).json()
    assert pending["status"] == "waiting_for_approval"
    assert pending["pending_action"]["element_id"] == send_id

    status = client.get(
        "/api/browser/status", params={"session_id": browser_session}
    ).json()
    assert status["status"] == "waiting_for_approval"

    denied = client.post("/api/browser/action", json={
        "session_id": browser_session,
        "deny": True,
    }).json()
    assert denied["status"] == "denied"
