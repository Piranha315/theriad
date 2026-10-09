from fastapi.testclient import TestClient

from theriad.api.server import app


def test_canonical_browser_session_flow_and_approval():
    client = TestClient(app)
    created = client.post("/api/browser/session/start", json={"goal": "submit a contact form"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    # The mock advances on observations. Polling the canonical endpoint exposes
    # a single normalized waiting_for_approval status.
    for _ in range(8):
        session = client.get(f"/api/browser/session/{session_id}").json()
        # Avoid a wall-clock wait in the unit test while exercising the same store.
        from theriad_browser.browser_agent import STORE
        STORE.get(session_id)._last_tick = 0
    session = client.get(f"/api/browser/session/{session_id}").json()
    assert session["status"] == "waiting_for_approval"
    assert session["pending_approval"]["proposed_action"] == "submit_form"

    approved = client.post(f"/api/browser/session/{session_id}/approve")
    assert approved.json()["status"] == "completed"


def test_built_react_product_is_the_root_when_available():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert 'id="root"' in response.text
