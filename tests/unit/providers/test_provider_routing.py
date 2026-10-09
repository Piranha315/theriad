from theriad.core.providers.routing import build_provider_routing_preferences


def test_economy_profile_sorts_by_price():
    route = build_provider_routing_preferences("xiaomi/mimo-v2.5", speed=10, quality=50)
    assert route.to_openrouter()["sort"] == "price"
    assert "order" not in route.to_openrouter()


def test_fast_mimo_profile_uses_benchmarked_order_with_fallbacks():
    route = build_provider_routing_preferences("xiaomi/mimo-v2.5", speed=90, quality=50)
    payload = route.to_openrouter()
    assert payload["order"] == ["novita", "xiaomi", "parasail"]
    assert payload["allow_fallbacks"] is True
    assert payload["preferred_min_throughput"] > 70


def test_fast_unknown_model_uses_throughput_sort():
    route = build_provider_routing_preferences("vendor/new-model", speed=90, quality=50)
    assert route.to_openrouter()["sort"] == "throughput"
