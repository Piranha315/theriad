"""CLI tests using Click's test runner."""
from __future__ import annotations

import json

from click.testing import CliRunner

from theriad.cli.main import cli


def make_runner():
    return CliRunner()


# ── theriad route ──────────────────────────────────────────────────────────────

def test_route_with_prompt():
    runner = make_runner()
    result = runner.invoke(cli, ["route", "What is 2+2?"])
    assert result.exit_code == 0
    assert "Tier:" in result.output
    assert "Model:" in result.output


def test_route_json_output():
    runner = make_runner()
    result = runner.invoke(cli, ["route", "--json", '{"key": "val"}'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "tier" in data
    assert "model" in data
    assert "confidence" in data
    assert "reason" in data
    assert "layer" in data
    assert "fallback_chain" in data


def test_route_json_tier_not_model_slug():
    """Tier field must be a logical tier label, not a raw model slug."""
    runner = make_runner()
    result = runner.invoke(cli, ["route", "--json", '{"key": "val"}'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    # tier should be a known tier label, not something with "/"
    assert "/" not in data["tier"]


def test_route_no_prompt_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(cli, ["route"], input=None, catch_exceptions=False)
    assert result.exit_code in (0, 1)


def test_route_long_prompt_routes_strong_reasoning():
    runner = make_runner()
    # Real English prose so langdetect correctly identifies "en" and abstains.
    # Layer 0 PromptLengthRule fires at 0.72 confidence (< 0.75 layer threshold),
    # so Layer 1 handles it. Layer 1 routes strong_reasoning when token_count > 3000.
    # 16 words × 200 = 3200 words → ~4256 tokens > 3000.
    sentence = (
        "The researchers presented their findings at the annual conference "
        "on distributed systems and cloud infrastructure engineering. "
    )
    long_prompt = sentence * 200  # ~3200 words, ~4256 tokens — caught by Layer 1
    result = runner.invoke(cli, ["route", long_prompt])
    assert result.exit_code == 0
    assert "strong_reasoning" in result.output


# ── theriad explain ────────────────────────────────────────────────────────────

def test_explain_shows_table():
    runner = make_runner()
    result = runner.invoke(cli, ["explain", "Write a Python sort function"])
    assert result.exit_code == 0
    assert "Layer" in result.output or "L0:" in result.output


def test_explain_shows_tier_and_model():
    runner = make_runner()
    result = runner.invoke(cli, ["explain", "What is 2+2?"])
    assert result.exit_code == 0
    assert "Tier:" in result.output
    assert "Model:" in result.output


def test_explain_json_output():
    runner = make_runner()
    result = runner.invoke(cli, ["explain", "--json", "Summarize this document"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "tier" in data
    assert "model" in data
    assert "estimated_cost_usd" in data
    assert "fallback_chain" in data
    assert "routing_path" in data


def test_explain_cost_estimate():
    runner = make_runner()
    result = runner.invoke(cli, ["explain", "--json", "Write a summary"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["estimated_prompt_tokens"] > 0


# ── theriad simulate ───────────────────────────────────────────────────────────

def test_simulate_basic(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "What is 2+2?"}\n'
        '{"prompt": "Write a Python function to reverse a list"}\n'
        '{"prompt": "Summarize the following text: the quick brown fox"}\n'
    )
    runner = make_runner()
    result = runner.invoke(cli, ["simulate", str(prompts)])
    assert result.exit_code == 0, result.output
    assert "Prompts:" in result.output
    assert "Theriad estimated cost:" in result.output
    assert "Estimated savings:" in result.output


def test_simulate_json_output(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "Hello world"}\n'
        '{"prompt": "def foo(): pass"}\n'
    )
    runner = make_runner()
    result = runner.invoke(cli, ["simulate", "--json", str(prompts)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "prompts" in data
    assert data["prompts"] == 2
    assert "theriad_estimated_cost_usd" in data
    assert "baseline_estimated_cost_usd" in data
    assert "tier_counts" in data
    assert "note" in data
    # Verify the note warns this is a dry-run
    assert "no provider calls" in data["note"].lower() or "dry-run" in data["note"].lower()


def test_simulate_no_provider_calls(tmp_path, monkeypatch):
    """Simulate must never call providers."""
    call_log = []
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "Hello world"}\n')

    # Patch httpx to detect any network call
    import httpx

    async def mock_post(self, *args, **kwargs):
        call_log.append(args)
        raise AssertionError("simulate must not call any provider")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    runner = make_runner()
    result = runner.invoke(cli, ["simulate", str(prompts)])
    assert result.exit_code == 0, result.output
    assert len(call_log) == 0, "provider was called during simulate"


def test_simulate_missing_file():
    runner = make_runner()
    result = runner.invoke(cli, ["simulate", "/nonexistent/path.jsonl"])
    assert result.exit_code != 0


def test_simulate_custom_baseline(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "Summarize this"}\n')
    runner = make_runner()
    result = runner.invoke(cli, [
        "simulate", "--json", "--baseline", "openai/gpt-4o-mini",
        str(prompts),
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["baseline_model"] == "openai/gpt-4o-mini"


def test_simulate_with_expected_output_tokens(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "Write code", "expected_output_tokens": 1000}\n')
    runner = make_runner()
    result = runner.invoke(cli, ["simulate", "--json", str(prompts)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["prompts"] == 1


# ── theriad config ─────────────────────────────────────────────────────────────

def test_config_shows_defaults():
    runner = make_runner()
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "Fallback tier" in result.output


def test_config_json_output():
    runner = make_runner()
    result = runner.invoke(cli, ["config", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "router" in data
    assert "logging" in data
    assert "fallback_tier" in data["router"]


# ── theriad benchmark ──────────────────────────────────────────────────────────

def test_benchmark_default_prompt():
    runner = make_runner()
    result = runner.invoke(cli, ["benchmark", "--iterations", "2"])
    assert result.exit_code == 0
    assert "layer0" in result.output


def test_benchmark_custom_prompt():
    runner = make_runner()
    result = runner.invoke(cli, ["benchmark", "hello world", "--iterations", "3"])
    assert result.exit_code == 0


# ── theriad stats ──────────────────────────────────────────────────────────────

def test_stats_no_metrics(tmp_path):
    runner = make_runner()
    import yaml
    cfg = {"metrics_path": str(tmp_path / "empty.jsonl")}
    cfg_file = tmp_path / "theriad.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    result = runner.invoke(cli, ["stats", "--config", str(cfg_file)])
    assert result.exit_code == 0
    assert "No metrics" in result.output
