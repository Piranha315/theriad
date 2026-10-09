from __future__ import annotations

import json
from io import BytesIO

import pytest

from theriad.routing.openrouter_metadata import (
    OpenRouterMetadataError,
    compare_metadata_to_plan,
    fetch_openrouter_metadata,
)


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _catalog(*, unrelated_name: str, prompt_price: str = "0") -> bytes:
    return json.dumps(
        {
            "data": [
                {
                    "id": "model/free",
                    "name": "Reviewed model",
                    "pricing": {"prompt": prompt_price, "completion": "0"},
                    "supported_parameters": ["tools", "tool_choice", "temperature"],
                    "architecture": {"modality": "text->text"},
                    "top_provider": {"context_length": 4096},
                },
                {
                    "id": "unrelated/model",
                    "name": unrelated_name,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": [],
                },
            ]
        },
        sort_keys=True,
    ).encode()


def test_unrelated_catalog_changes_do_not_invalidate_reviewed_models(monkeypatch):
    payloads = iter((_catalog(unrelated_name="Before"), _catalog(unrelated_name="After")))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response(next(payloads))
    )

    before = fetch_openrouter_metadata(["model/free"], tool_model_ids=["model/free"])
    after = fetch_openrouter_metadata(["model/free"], tool_model_ids=["model/free"])

    assert before["response_sha256"] == after["response_sha256"]
    assert before["catalog_response_sha256"] != after["catalog_response_sha256"]
    compare_metadata_to_plan(
        after,
        {"model/free": {"input_per_1m": 0.0, "output_per_1m": 0.0}},
        expected_response_sha256=before["response_sha256"],
    )


def test_relevant_price_change_still_fails_closed(monkeypatch):
    payloads = iter(
        (
            _catalog(unrelated_name="Same"),
            _catalog(unrelated_name="Same", prompt_price="0.000001"),
        )
    )
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response(next(payloads))
    )

    before = fetch_openrouter_metadata(["model/free"], tool_model_ids=["model/free"])
    after = fetch_openrouter_metadata(["model/free"], tool_model_ids=["model/free"])

    with pytest.raises(OpenRouterMetadataError, match="relevant OpenRouter model metadata changed"):
        compare_metadata_to_plan(
            after,
            {"model/free": {"input_per_1m": 0.0, "output_per_1m": 0.0}},
            expected_response_sha256=before["response_sha256"],
        )
