"""
End-to-end tests for image attachments.

Covers the path the user actually exercises: drop an image into chat, have it
reach a vision-capable model as an image content block rather than as mangled
text or not at all.

No provider is called — `complete` is mocked and the captured request is
inspected, which is the only way to assert what would have gone over the wire.
"""
from __future__ import annotations

import base64
import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import theriad.api.projects as projects_module
import theriad.api.server as server_module
from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.storage import database as db_module
from theriad.workflow.detector import WorkflowDetector

# A 1x1 transparent PNG — real bytes, so nothing here depends on a fake decoder.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _mock_wf_detector():
    return WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    db_module.close_db()
    db_module.configure(db_path)
    projects_module.set_upload_base(upload_dir)
    yield db_path, upload_dir
    db_module.close_db()


@pytest.fixture()
def client(monkeypatch, tmp_db):
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
    return TestClient(server_module.app)


def _upload_image(client, name="screenshot.png", data=PNG_BYTES):
    response = client.post(
        "/api/files", files={"file": (name, io.BytesIO(data), "image/png")}
    )
    assert response.status_code == 201, response.text
    return response.json()


# ── Upload ────────────────────────────────────────────────────────────────────

class TestImageUpload:

    def test_png_upload_is_accepted(self, client):
        saved = _upload_image(client)
        assert saved["original_name"] == "screenshot.png"
        assert saved["size_bytes"] == len(PNG_BYTES)

    @pytest.mark.parametrize(
        "name,mime",
        [
            ("a.png", "image/png"),
            ("b.jpg", "image/jpeg"),
            ("c.jpeg", "image/jpeg"),
            ("d.gif", "image/gif"),
            ("e.webp", "image/webp"),
        ],
    )
    def test_all_image_types_accepted(self, client, name, mime):
        response = client.post(
            "/api/files", files={"file": (name, io.BytesIO(PNG_BYTES), mime)}
        )
        assert response.status_code == 201, response.text

    def test_image_bytes_survive_round_trip(self, client, tmp_db):
        """Regression: images used to be utf-8 decoded, destroying the bytes."""
        saved = _upload_image(client)
        _db_path, upload_dir = tmp_db
        stored = (upload_dir / "_standalone" / saved["filename"]).read_bytes()
        assert stored == PNG_BYTES

    def test_image_is_not_text_extracted(self, client):
        """An image decoded as text would inject replacement chars into the prompt."""
        saved = _upload_image(client)
        assert projects_module.get_file_texts([saved["id"]]) == []

    def test_oversized_image_rejected(self, client):
        oversized = b"\x89PNG" + b"\x00" * (projects_module.MAX_IMAGE_BYTES + 1)
        response = client.post(
            "/api/files", files={"file": ("big.png", io.BytesIO(oversized), "image/png")}
        )
        assert response.status_code == 413

    def test_still_rejects_genuinely_unsupported_types(self, client):
        response = client.post(
            "/api/files",
            files={"file": ("a.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        )
        assert response.status_code == 415


# ── Data URI construction ─────────────────────────────────────────────────────

class TestGetFileImages:

    def test_returns_data_uri_with_correct_mime(self, client):
        saved = _upload_image(client)
        images = projects_module.get_file_images([saved["id"]])
        assert len(images) == 1
        name, uri = images[0]
        assert name == "screenshot.png"
        assert uri.startswith("data:image/png;base64,")

    def test_data_uri_decodes_back_to_original_bytes(self, client):
        saved = _upload_image(client)
        _name, uri = projects_module.get_file_images([saved["id"]])[0]
        assert base64.b64decode(uri.split(",", 1)[1]) == PNG_BYTES

    def test_jpg_gets_jpeg_mime(self, client):
        response = client.post(
            "/api/files", files={"file": ("p.jpg", io.BytesIO(PNG_BYTES), "image/jpeg")}
        )
        _name, uri = projects_module.get_file_images([response.json()["id"]])[0]
        assert uri.startswith("data:image/jpeg;base64,")

    def test_mime_comes_from_extension_not_client_claim(self, client):
        """Content type is client-controlled; the extension is what we stored."""
        response = client.post(
            "/api/files",
            files={"file": ("real.png", io.BytesIO(PNG_BYTES), "application/x-lying")},
        )
        _name, uri = projects_module.get_file_images([response.json()["id"]])[0]
        assert uri.startswith("data:image/png;base64,")

    def test_text_attachments_are_not_returned_as_images(self, client):
        response = client.post(
            "/api/files", files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")}
        )
        assert projects_module.get_file_images([response.json()["id"]]) == []

    def test_empty_input_returns_empty(self):
        assert projects_module.get_file_images([]) == []

    def test_missing_bytes_are_skipped_not_fatal(self, client, tmp_db):
        saved = _upload_image(client)
        _db_path, upload_dir = tmp_db
        (upload_dir / "_standalone" / saved["filename"]).unlink()
        assert projects_module.get_file_images([saved["id"]]) == []


# ── End to end through the chat endpoint ──────────────────────────────────────

class TestChatWithImage:

    def _send(self, client, file_ids, message="what is in this image?"):
        captured = {}

        async def _fake_complete(request):
            captured["request"] = request
            from theriad.core.providers.interfaces import CompletionResponse

            return CompletionResponse(
                model=request.model,
                content="A single transparent pixel.",
                prompt_tokens=10,
                completion_tokens=8,
                latency_ms=100.0,
                provider="openrouter",
            )

        with patch.object(server_module, "_resolve_assistant_api_key", return_value="k"):
            with patch(
                "theriad.core.providers.factory.build_provider"
            ) as build:
                provider = build.return_value
                provider.name = "openrouter"
                provider.complete = AsyncMock(side_effect=_fake_complete)
                response = client.post(
                    "/api/chat/send",
                    json={
                        "message": message,
                        "mode": "execute",
                        "allow_paid_calls": True,
                        "file_ids": file_ids,
                    },
                )
        return response, captured

    def test_image_reaches_the_model_as_an_image_block(self, client):
        saved = _upload_image(client)
        response, captured = self._send(client, [saved["id"]])
        assert response.status_code == 200, response.text

        user_msg = [m for m in captured["request"].messages if m.role == "user"][0]
        assert user_msg.has_images, "image attachment never became an image block"
        image_blocks = [
            b for b in user_msg.content if b.get("type") == "image_url"
        ]
        assert len(image_blocks) == 1
        assert image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_prompt_text_is_preserved_alongside_the_image(self, client):
        saved = _upload_image(client)
        _response, captured = self._send(client, [saved["id"]], "describe this")
        user_msg = [m for m in captured["request"].messages if m.role == "user"][0]
        assert "describe this" in user_msg.text

    def test_request_is_routed_to_a_vision_capable_model(self, client):
        from theriad.models.vision import supports_vision

        saved = _upload_image(client)
        _response, captured = self._send(client, [saved["id"]])
        assert supports_vision(captured["request"].model), (
            f"routed to {captured['request'].model}, which cannot see images"
        )

    # Default routing happens to land on a vision-capable model, so asserting
    # "the chosen model can see" does not by itself prove the gate works. These
    # two force a text-only candidate list — the case the gate exists for.
    TEXT_ONLY = [
        ("openrouter", "qwen/qwen3.6-plus"),
        ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ]

    def test_gate_overrides_a_candidate_list_that_cannot_see(self, client):
        from theriad.models.vision import supports_vision

        saved = _upload_image(client)
        with patch.object(
            server_module, "_assistant_candidates", return_value=list(self.TEXT_ONLY)
        ):
            _response, captured = self._send(client, [saved["id"]])
        chosen = captured["request"].model
        assert supports_vision(chosen), (
            f"image request routed to {chosen}, which cannot see images"
        )

    def test_gate_stays_out_of_the_way_without_an_image(self, client):
        """The gate must only engage for image requests, not reroute everything."""
        with patch.object(
            server_module, "_assistant_candidates", return_value=list(self.TEXT_ONLY)
        ):
            _response, captured = self._send(client, [])
        assert captured["request"].model == "qwen/qwen3.6-plus"

    def test_text_only_request_still_sends_a_plain_string(self, client):
        """The common path must be byte-identical to before this feature."""
        _response, captured = self._send(client, [], "just text")
        user_msg = [m for m in captured["request"].messages if m.role == "user"][0]
        assert isinstance(user_msg.content, str)
        assert user_msg.has_images is False

    def test_system_message_stays_a_plain_string(self, client):
        saved = _upload_image(client)
        _response, captured = self._send(client, [saved["id"]])
        system = [m for m in captured["request"].messages if m.role == "system"]
        assert system and isinstance(system[0].content, str)
