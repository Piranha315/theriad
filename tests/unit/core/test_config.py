"""Tests for config schema and loader."""
from theriad.core.config.loader import load_config
from theriad.core.config.schema import TheriadConfig, ProviderConfig, RouterConfig


def test_default_config_instantiates():
    cfg = TheriadConfig()
    assert cfg.router.min_confidence == 0.75
    assert cfg.router.fallback_tier == "cheap_bulk"
    assert cfg.router.policy == "balanced"


def test_router_layers_default():
    cfg = TheriadConfig()
    assert "layer0" in cfg.router.layers


def test_provider_config():
    p = ProviderConfig(name="openrouter", base_url="https://openrouter.ai/api/v1")
    assert p.timeout_seconds == 30.0
    assert p.max_retries == 3


def test_load_config_no_file():
    # Should return default config if no file found
    cfg = load_config("/nonexistent/path.yaml")
    assert isinstance(cfg, TheriadConfig)


def test_router_config_policy_field():
    cfg = RouterConfig()
    assert cfg.policy in {"balanced", "aggressive_savings", "quality_first"}


def test_full_config_round_trip(tmp_path):
    import yaml
    data = {
        "providers": {
            "openrouter": {
                "name": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            }
        },
        "router": {
            "min_confidence": 0.8,
            "fallback_tier": "cheap_bulk",
            "layers": ["layer0"],
            "policy": "aggressive_savings",
        },
    }
    cfg_file = tmp_path / "theriad.yaml"
    cfg_file.write_text(yaml.dump(data))
    cfg = load_config(cfg_file)
    assert cfg.router.min_confidence == 0.8
    assert cfg.router.fallback_tier == "cheap_bulk"
    assert cfg.router.policy == "aggressive_savings"
    assert "openrouter" in cfg.providers
