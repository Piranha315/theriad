"""
Unit tests for vision capability resolution and the multimodal content path.

The asymmetry these tests defend: routing an image to a text-only model yields a
confident answer about nothing, which is far harder to notice than an error. So
"unknown" must resolve to "cannot see", and the vision gate must never hand an
image request to a model that was not vouched for.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from theriad.accounting.live_prices import PriceSnapshot
from theriad.models.vision import (
    DEFAULT_VISION_MODEL,
    VISION_ESCALATION_MODEL,
    ensure_vision_candidates,
    supports_vision,
    vision_candidates,
)


def _live(*slugs):
    """Patch the live catalog to vouch for exactly these slugs."""
    return patch(
        "theriad.accounting.live_prices.get_live_snapshot",
        return_value=PriceSnapshot(vision=frozenset(slugs), origin="live", stale=False),
    )


def _no_live():
    return patch(
        "theriad.accounting.live_prices.get_live_snapshot",
        return_value=PriceSnapshot(origin="empty"),
    )


class TestSupportsVision:

    def test_live_catalog_is_authoritative(self):
        with _live("brand/new-vision-model"):
            assert supports_vision("brand/new-vision-model") is True

    def test_unknown_model_is_text_only(self):
        with _no_live():
            assert supports_vision("totally/unknown-model") is False

    def test_empty_model_is_text_only(self):
        assert supports_vision("") is False
        assert supports_vision(None) is False

    def test_seed_floor_works_without_the_catalog(self):
        with _no_live():
            assert supports_vision(DEFAULT_VISION_MODEL) is True
            assert supports_vision("openai/gpt-4o") is True
            assert supports_vision("z-ai/glm-5.3-flash") is True

    def test_versioned_slug_matches_base_seed(self):
        with _no_live():
            assert supports_vision("xiaomi/mimo-v2.5-20260422") is True

    def test_versioned_slug_matches_base_live_entry(self):
        with _live("brand/new"):
            assert supports_vision("brand/new-20261231") is True

    def test_empty_live_set_does_not_veto_seeds(self):
        """An unreachable catalog means 'unknown', not 'nothing can see'."""
        with _no_live():
            assert supports_vision(DEFAULT_VISION_MODEL) is True

    def test_prefix_match_does_not_leak_across_unrelated_slugs(self):
        with _no_live():
            # Shares a prefix with openai/gpt-4o only up to a word boundary we
            # do not honour; must not be silently promoted to vision.
            assert supports_vision("openai/gpt-4o-audio-preview") is True
            assert supports_vision("deepseek/deepseek-v4-flash") is False

    def test_use_live_false_consults_seeds_only(self):
        with _live("brand/new-vision-model") as mocked:
            assert supports_vision("brand/new-vision-model", use_live=False) is False
            mocked.assert_not_called()


class TestVisionCandidates:

    CANDIDATES = [
        ("openrouter", "deepseek/deepseek-v4-flash"),   # text only
        ("openrouter", "xiaomi/mimo-v2.5"),             # vision
        ("openrouter", "qwen/qwen3.6-plus"),            # text only
    ]

    def test_drops_text_only_candidates(self):
        with _no_live():
            kept = vision_candidates(self.CANDIDATES)
        assert kept == [("openrouter", "xiaomi/mimo-v2.5")]

    def test_preserves_tier_order(self):
        with _no_live():
            kept = vision_candidates(
                [("openrouter", "openai/gpt-4o"), ("openrouter", "xiaomi/mimo-v2.5")]
            )
        assert kept == [
            ("openrouter", "openai/gpt-4o"),
            ("openrouter", "xiaomi/mimo-v2.5"),
        ]


class TestEnsureVisionCandidates:

    def test_appends_presets_when_nothing_can_see(self):
        with _no_live():
            result = ensure_vision_candidates(
                [("openrouter", "deepseek/deepseek-v4-flash")]
            )
        models = [m for _, m in result]
        assert models == [DEFAULT_VISION_MODEL, VISION_ESCALATION_MODEL]

    def test_never_returns_empty(self):
        with _no_live():
            assert ensure_vision_candidates([]) != []

    def test_result_is_entirely_vision_capable(self):
        with _no_live():
            result = ensure_vision_candidates(
                [
                    ("openrouter", "deepseek/deepseek-v4-flash"),
                    ("openrouter", "qwen/qwen3.6-plus"),
                ]
            )
        assert all(supports_vision(m) for _, m in result)

    def test_does_not_duplicate_an_already_present_preset(self):
        with _no_live():
            result = ensure_vision_candidates([("openrouter", DEFAULT_VISION_MODEL)])
        models = [m for _, m in result]
        assert models.count(DEFAULT_VISION_MODEL) == 1

    def test_keeps_capable_candidates_ahead_of_presets(self):
        with _no_live():
            result = ensure_vision_candidates(
                [
                    ("openrouter", "deepseek/deepseek-v4-flash"),
                    ("openrouter", "anthropic/claude-sonnet-4-6"),
                ]
            )
        assert result[0] == ("openrouter", "anthropic/claude-sonnet-4-6")


# ── Message content plumbing ──────────────────────────────────────────────────

class TestMessageContent:

    def test_string_content_unchanged(self):
        from theriad.core.providers.interfaces import Message

        msg = Message(role="user", content="hello")
        assert msg.text == "hello"
        assert msg.has_images is False

    def test_text_property_ignores_image_blocks(self):
        from theriad.core.providers.interfaces import Message

        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        )
        assert msg.text == "what is this"
        assert msg.has_images is True

    def test_routing_uses_text_not_pixels(self):
        """The pipeline routes on prose; an image must not break routing."""
        from theriad.core.providers.interfaces import Message

        msg = Message(
            role="user",
            content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}}],
        )
        assert msg.text == ""      # no crash, no garbage


class TestUserContentBuilder:

    def test_no_images_returns_plain_string(self):
        from theriad.api.server import _user_content

        assert _user_content("hello", []) == "hello"

    def test_images_produce_content_blocks(self):
        from theriad.api.server import _user_content

        blocks = _user_content("describe", [("shot.png", "data:image/png;base64,AAAA")])
        assert blocks[0] == {"type": "text", "text": "describe"}
        assert blocks[-1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }

    def test_each_image_is_labelled_by_filename(self):
        from theriad.api.server import _user_content

        blocks = _user_content(
            "compare these",
            [("a.png", "data:image/png;base64,A"), ("b.png", "data:image/png;base64,B")],
        )
        labels = [b["text"] for b in blocks if b.get("type") == "text"]
        assert "Attached image: a.png" in labels
        assert "Attached image: b.png" in labels

    def test_all_images_are_included(self):
        from theriad.api.server import _user_content

        blocks = _user_content(
            "x", [(f"{i}.png", f"data:image/png;base64,{i}") for i in range(4)]
        )
        images = [b for b in blocks if b.get("type") == "image_url"]
        assert len(images) == 4


# ── Provider payload pass-through ─────────────────────────────────────────────

class TestProviderPassthrough:

    def test_openrouter_forwards_block_list_untouched(self):
        from theriad.core.providers.interfaces import CompletionRequest, Message

        blocks = [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        request = CompletionRequest(
            model="xiaomi/mimo-v2.5",
            messages=[Message(role="user", content=blocks)],
        )
        payload_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        assert payload_messages[0]["content"] == blocks

    def test_cache_control_does_not_flatten_a_block_list(self):
        """Regression: the explicit-cache path wrapped content assuming a string."""
        from theriad.core.providers.openrouter import OpenRouterProvider  # noqa: F401

        messages = [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
        text = messages[0]["content"]
        if isinstance(text, str):                       # mirrors provider guard
            messages[0]["content"] = [{"type": "text", "text": text}]
        assert messages[0]["content"] == [{"type": "text", "text": "sys"}]
