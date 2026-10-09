from types import SimpleNamespace

from theriad.api.server import _provider_request_extra


def _preferences():
    return SimpleNamespace(speed=50, quality=50)


def test_openrouter_model_hint_keeps_three_candidates():
    extra = _provider_request_extra(
        "openrouter",
        "model/primary",
        _preferences(),
        ["model/primary", "model/fallback-one", "model/fallback-two"],
    )

    assert extra["models"] == [
        "model/primary",
        "model/fallback-one",
        "model/fallback-two",
    ]


def test_openrouter_model_hint_is_omitted_for_longer_server_fallback_chains():
    extra = _provider_request_extra(
        "openrouter",
        "model/primary",
        _preferences(),
        [
            "model/primary",
            "model/fallback-one",
            "model/fallback-two",
            "model/fallback-three",
        ],
    )

    assert "models" not in extra
    assert extra["provider"]["sort"] == "price"


def test_chat_attempt_can_pin_one_candidate_without_server_side_fallback():
    extra = _provider_request_extra(
        "openrouter",
        "model/primary",
        _preferences(),
        ["model/primary"],
    )

    assert "models" not in extra
    assert extra["provider"]["sort"] == "price"
