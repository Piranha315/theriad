"""
Tests for the Harness Integration API (/v1/chat/completions, /v1/models).

All provider calls are mocked — no real API key or network access required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import theriad.api.server as server_module


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_wf_detector():
    from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
    from theriad.workflow.detector import WorkflowDetector
    return WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))


def _fake_completion(
    content="Test response from Theriad.",
    model="qwen/qwen3-14b",
    provider="openrouter",
    prompt_tokens=15,
    completion_tokens=8,
):
    from theriad.core.providers.interfaces import CompletionResponse
    return CompletionResponse(
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=210.0,
        provider=provider,
        finish_reason="stop",
    )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
    return TestClient(server_module.app)


@pytest.fixture()
def client_with_key(monkeypatch):
    """TestClient with a dummy configured API key so auth doesn't block tests."""
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
    monkeypatch.setattr(
        "theriad.settings.secrets.get_openrouter_api_key",
        lambda: "sk-or-test-key",
    )
    return TestClient(server_module.app)


# ── GET /v1/models ────────────────────────────────────────────────────────────

class TestListModels:

    def test_returns_list_object(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

    def test_includes_theriad_virtual_models(self, client):
        r = client.get("/v1/models")
        ids = {m["id"] for m in r.json()["data"]}
        assert "theriad" in ids
        assert "theriad" in ids
        assert "theriad/fast" in ids
        assert "theriad/quality" in ids
        assert "theriad/free" in ids

    def test_includes_tier_names(self, client):
        r = client.get("/v1/models")
        ids = {m["id"] for m in r.json()["data"]}
        assert "cheap_bulk" in ids
        assert "code_strong" in ids
        assert "premium" in ids

    def test_owned_by_theriad(self, client):
        r = client.get("/v1/models")
        for model in r.json()["data"]:
            assert model["owned_by"] == "theriad"
            assert model["object"] == "model"


# ── POST /v1/chat/completions — validation ────────────────────────────────────

class TestChatCompletionsValidation:

    def test_empty_messages_returns_422(self, client_with_key):
        r = client_with_key.post(
            "/v1/chat/completions",
            json={"model": "theriad", "messages": []},
        )
        assert r.status_code == 422
        assert "messages" in r.json()["error"]["message"]

    def test_streaming_returns_501(self, client_with_key):
        r = client_with_key.post(
            "/v1/chat/completions",
            json={
                "model": "theriad",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 501
        assert "streaming" in r.json()["error"]["message"].lower()

    def test_missing_api_key_returns_401(self, monkeypatch):
        from fastapi.testclient import TestClient
        monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
        monkeypatch.setattr("theriad.settings.secrets.get_openrouter_api_key", lambda: "")
        c = TestClient(server_module.app)
        r = c.post(
            "/v1/chat/completions",
            json={"model": "theriad", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 401

    def test_bearer_token_used_as_api_key(self, monkeypatch):
        monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
        # No configured key
        monkeypatch.setattr("theriad.settings.secrets.get_openrouter_api_key", lambda: "")
        c = TestClient(server_module.app)

        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion())
            mock_build.return_value = mock_provider

            r = c.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-from-header"},
                json={"model": "theriad", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 200
        # Verify the key from the header was passed to build_provider
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-or-from-header"


# ── POST /v1/chat/completions — happy paths ───────────────────────────────────

class TestChatCompletionsRouting:

    def _post(self, client, model="theriad", extra=None):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Write a quick sort in Python."}],
        }
        if extra:
            payload.update(extra)
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion())
            mock_build.return_value = mock_provider
            return client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json=payload,
            ), mock_build

    def test_returns_openai_shape(self, client):
        r, _ = self._post(client)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("theriad-")
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0

    def test_x_theriad_metadata_present(self, client):
        r, _ = self._post(client)
        xf = r.json()["x_theriad"]
        assert "tier" in xf
        assert "provider" in xf
        assert "model" in xf
        assert "routing_layer" in xf
        assert "routing_reason" in xf
        assert "confidence" in xf

    def test_renderable_request_injects_renderer_contract(self, client):
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion("```html\n<h1>Preview</h1>\n```"))
            mock_build.return_value = mock_provider
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={
                    "model": "theriad",
                    "messages": [{"role": "user", "content": "Design a website for me."}],
                },
            )

        assert response.status_code == 200
        request = mock_provider.complete.await_args.args[0]
        assert any(
            message.role == "system" and "RENDERABLE_ARTIFACT_OUTPUT" in message.content
            for message in request.messages
        )
        renderer = response.json()["x_theriad"]["renderer"]
        assert renderer["enabled"] is True
        assert renderer["artifact_format"] == "self_contained_html"

    def test_model_theriad_fast_sets_speed_pref(self, client):
        with patch("theriad.models.policy.PolicyResolver.resolve") as mock_resolve:
            from theriad.models.policy import PolicyResolution
            from theriad.models.tiers import ModelCandidate
            candidate = ModelCandidate(
                tier="cheap_bulk", provider="openrouter", model="qwen/qwen3-14b",
                role="primary", input_cost_per_million=0.1, output_cost_per_million=0.1,
                context_window=128000, quality_prior=0.75,
            )
            mock_resolve.return_value = PolicyResolution(
                tier="cheap_bulk", selected=candidate, fallback_chain=[], policy="balanced",
            )
            with patch("theriad.core.providers.factory.build_provider") as mock_build:
                mock_provider = AsyncMock()
                mock_provider.complete = AsyncMock(return_value=_fake_completion())
                mock_build.return_value = mock_provider
                r = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-or-test"},
                    json={
                        "model": "theriad/fast",
                        "messages": [{"role": "user", "content": "quick task"}],
                    },
                )
        assert r.status_code == 200
        # speed_preference=80 should be passed to PolicyResolver
        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs.get("speed_preference") == 80

    def test_tier_pinned_model_skips_routing_layers(self, client):
        with patch("theriad.layers.layer0.layer.build_default_layer") as mock_l0:
            with patch("theriad.core.providers.factory.build_provider") as mock_build:
                mock_provider = AsyncMock()
                mock_provider.complete = AsyncMock(return_value=_fake_completion())
                mock_build.return_value = mock_provider
                r = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-or-test"},
                    json={
                        "model": "cheap_code",
                        "messages": [{"role": "user", "content": "fix this bug"}],
                    },
                )
        assert r.status_code == 200
        # Layer 0 should not have been called for a pinned tier
        mock_l0.assert_not_called()
        xf = r.json()["x_theriad"]
        assert xf["routing_layer"] == "pinned"

    def test_router_v2_on_skips_routing_layers(self, client):
        from types import SimpleNamespace

        from theriad.models.policy import PolicyResolution
        from theriad.models.registry import DEFAULT_REGISTRY

        chosen = DEFAULT_REGISTRY.candidates_for_tier("premium")[0]
        decision = SimpleNamespace(router_version="test-v2", mode="on", applied=True, lane="general",
                                   knee_threshold=0.27, reason="cheapest_above_knee", ood_score=0.1,
                                   shadow_challenger=None)
        learned = PolicyResolution(tier="premium", selected=chosen, fallback_chain=[], policy="router_v2",
                                   reason_for_stay_or_switch="router_v2:cheapest_above_knee",
                                   router_version="test-v2", router_v2_decision=decision)
        with patch("theriad.routing.v2.learned_only_active", return_value=True), \
             patch("theriad.models.policy.PolicyResolver.resolve_learned", return_value=learned), \
             patch("theriad.layers.layer0.layer.build_default_layer") as mock_l0, \
             patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion(model=chosen.model))
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={"model": "theriad/quality", "messages": [{"role": "user", "content": "write a story"}]},
            )
        assert r.status_code == 200
        mock_l0.assert_not_called()
        xf = r.json()["x_theriad"]
        assert xf["routing_layer"] == "router_v2"
        assert xf["router_v2"]["applied"] is True
        assert xf["router_v2"]["router_version"] == "test-v2"

    def test_json_response_format_skips_renderer_and_is_forwarded(self, client):
        prompt = "Create an interactive HTML dashboard page with charts"
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion(content='{"ok": true}'))
            mock_build.return_value = mock_provider
            plain = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={"model": "cheap_bulk", "messages": [{"role": "user", "content": prompt}]},
            )
            as_json = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={
                    "model": "cheap_bulk",
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        assert plain.status_code == 200 and as_json.status_code == 200
        assert "renderer" in plain.json()["x_theriad"]
        assert "renderer" not in as_json.json()["x_theriad"]
        req = mock_provider.complete.call_args_list[-1].args[0]
        assert req.extra["response_format"] == {"type": "json_object"}
        assert req.temperature == 0.7
        assert [m.content for m in req.messages] == [prompt]

    def test_direct_model_slug_routes_to_openrouter(self, client):
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(
                return_value=_fake_completion(model="anthropic/claude-haiku-4-5")
            )
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={
                    "model": "anthropic/claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert r.status_code == 200
        xf = r.json()["x_theriad"]
        assert xf["routing_layer"] == "direct"
        assert xf["provider"] == "openrouter"

    def test_unknown_openai_fields_ignored(self, client):
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion())
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={
                    "model": "theriad",
                    "messages": [{"role": "user", "content": "hi"}],
                    "top_p": 0.9,          # OpenAI field not used by Theriad
                    "presence_penalty": 0.0,
                },
            )
        assert r.status_code == 200

    def test_content_block_array_flattened(self, client):
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=_fake_completion())
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={
                    "model": "theriad",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is 2 + 2?"},
                            ],
                        }
                    ],
                },
            )
        assert r.status_code == 200

    def test_x_theriad_extension_fields_override_defaults(self, client):
        with patch("theriad.models.policy.PolicyResolver.resolve") as mock_resolve:
            from theriad.models.policy import PolicyResolution
            from theriad.models.tiers import ModelCandidate
            candidate = ModelCandidate(
                tier="premium", provider="openrouter", model="openai/gpt-4o",
                role="primary", input_cost_per_million=2.5, output_cost_per_million=10.0,
                context_window=128000, quality_prior=0.95,
            )
            mock_resolve.return_value = PolicyResolution(
                tier="premium", selected=candidate, fallback_chain=[], policy="quality_first",
            )
            with patch("theriad.core.providers.factory.build_provider") as mock_build:
                mock_provider = AsyncMock()
                mock_provider.complete = AsyncMock(return_value=_fake_completion())
                mock_build.return_value = mock_provider
                r = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-or-test"},
                    json={
                        "model": "theriad",
                        "messages": [{"role": "user", "content": "complex analysis"}],
                        "x_theriad_policy": "quality_first",
                        "x_theriad_quality": 90,
                    },
                )
        assert r.status_code == 200
        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs.get("quality_preference") == 90

    def test_usage_counts_correct(self, client):
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(
                return_value=_fake_completion(prompt_tokens=42, completion_tokens=17)
            )
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={"model": "theriad", "messages": [{"role": "user", "content": "test"}]},
            )
        usage = r.json()["usage"]
        assert usage["prompt_tokens"] == 42
        assert usage["completion_tokens"] == 17
        assert usage["total_tokens"] == 59

    def test_theriad_alias_accepted(self, client):
        r, _ = self._post(client, model="theriad")
        assert r.status_code == 200

    def test_theriad_free_uses_moe_policy(self, client):
        with patch("theriad.models.policy.PolicyResolver.resolve") as mock_resolve:
            from theriad.models.policy import PolicyResolution
            from theriad.models.tiers import ModelCandidate
            candidate = ModelCandidate(
                tier="free_moe_general", provider="openrouter",
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                role="primary", input_cost_per_million=0.0, output_cost_per_million=0.0,
                context_window=131072, quality_prior=0.68,
            )
            mock_resolve.return_value = PolicyResolution(
                tier="free_moe_general", selected=candidate,
                fallback_chain=[], policy="free_moe_experimental",
            )
            with patch("theriad.core.providers.factory.build_provider") as mock_build:
                mock_provider = AsyncMock()
                mock_provider.complete = AsyncMock(return_value=_fake_completion())
                mock_build.return_value = mock_provider
                r = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-or-test"},
                    json={
                        "model": "theriad/free",
                        "messages": [{"role": "user", "content": "summarize this"}],
                    },
                )
        assert r.status_code == 200
        call_args = mock_resolve.call_args
        assert call_args.args[1] == "free_moe_experimental"


# ── Provider failure paths ────────────────────────────────────────────────────

class TestChatCompletionsFailures:

    def test_all_providers_fail_returns_502(self, client):
        from theriad.core.providers.interfaces import ProviderError
        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(
                side_effect=ProviderError("openrouter", "upstream error", status_code=503)
            )
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={"model": "theriad", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 502
        assert "error" in r.json()

    def test_empty_response_triggers_fallback(self, client):
        from theriad.core.providers.interfaces import CompletionResponse, ProviderError
        call_count = 0

        async def _side_effect(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: empty content
                return CompletionResponse(
                    model="some-model", content="", prompt_tokens=5,
                    completion_tokens=3, latency_ms=100.0, provider="openrouter",
                    finish_reason="stop",
                )
            return _fake_completion()

        with patch("theriad.core.providers.factory.build_provider") as mock_build:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(side_effect=_side_effect)
            mock_build.return_value = mock_provider
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-or-test"},
                json={"model": "theriad", "messages": [{"role": "user", "content": "hi"}]},
            )
        # Should succeed on the second attempt from the fallback chain
        assert r.status_code == 200
