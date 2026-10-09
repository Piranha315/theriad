"""Lightweight live evaluation for ``free_moe_experimental``.

Run with::

    python3 -m theriad.benchmark.free_moe_eval

Classification is always evaluated locally. Live response checks are skipped
with exit code 0 when ``OPENROUTER_API_KEY`` is not set.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from theriad.models.free_moe import FreeMoeResolver, route_free_moe


@dataclass(frozen=True)
class EvalPrompt:
    pool: str
    prompt: str


PROMPTS: tuple[EvalPrompt, ...] = (
    EvalPrompt("general_free", "What are three practical ways to build a morning routine?"),
    EvalPrompt("general_free", "Explain why leaves change color in autumn."),
    EvalPrompt("general_free", "Give me a creative dinner idea using chickpeas."),
    EvalPrompt("general_free", "How can a team make weekly meetings more useful?"),
    EvalPrompt("general_free", "Write a friendly two-sentence welcome message."),
    EvalPrompt("code_free", "Debug this TypeError in my Python list processing."),
    EvalPrompt("code_free", "Write a function that removes duplicate strings."),
    EvalPrompt("code_free", "Why does pytest fail to discover this test class?"),
    EvalPrompt("code_free", "```js\nconst value = missing.name\n```\nFix this bug."),
    EvalPrompt("code_free", "My Rust program hits a runtime error; explain how to trace it."),
    EvalPrompt("verify_free", "Fact-check the claim that lightning never strikes twice."),
    EvalPrompt("verify_free", "Is it true that humans use only ten percent of their brains?"),
    EvalPrompt("verify_free", "Verify this claim and identify what evidence would be needed."),
    EvalPrompt("verify_free", "According to the cited source, what conclusion is justified?"),
    EvalPrompt("verify_free", "Does drinking coffee cause dehydration?"),
    EvalPrompt("summarize_free", "Summarize these meeting notes into three decisions."),
    EvalPrompt("summarize_free", "Extract the key points from this product brief."),
    EvalPrompt("summarize_free", "Give me a TL;DR of the following article."),
    EvalPrompt("summarize_free", "Rewrite this memo as concise bullet points."),
    EvalPrompt("summarize_free", "A neutral document passage. " * 40),
)


def _call_openrouter(api_key: str, model: str, prompt: str) -> tuple[float, float, bool]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://theriad.ai/free-moe-eval",
            "X-Title": "Theriad free_moe_experimental eval",
        },
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=45.0) as response:
            result = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.monotonic() - started) * 1000
        usage = result.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
        return latency_ms, cost, bool(content)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError):
        return (time.monotonic() - started) * 1000, 0.0, False


def _print_row(
    pool: str,
    prompt: str,
    model: str,
    latency_ms: float | None,
    cost_usd: float | None,
    response_ok: bool | None,
) -> None:
    snippet = " ".join(prompt.split())[:48]
    latency = "skipped" if latency_ms is None else f"{latency_ms:.0f}"
    cost = "skipped" if cost_usd is None else f"{cost_usd:.8f}"
    ok = "skipped" if response_ok is None else str(response_ok).lower()
    print(f"{pool:<16} | {snippet:<48} | {model:<47} | {latency:>10} | {cost:>10} | {ok}")


def main() -> int:
    predictions = [(item, route_free_moe(item.prompt)) for item in PROMPTS]
    correct = sum(decision.pool == item.pool for item, decision in predictions)
    print(f"Classifier accuracy: {correct}/{len(PROMPTS)} ({correct / len(PROMPTS):.0%})")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY is not set; skipping live OpenRouter response checks.")
        return 0

    print(
        "pool             | prompt_snippet                                   "
        "     | model                                           | latency_ms | cost_usd   | response_ok"
    )
    print("-" * 155)

    resolver = FreeMoeResolver()
    for item, _decision in predictions:
        resolution = resolver.resolve(item.prompt)
        latency_ms, cost_usd, response_ok = _call_openrouter(
            api_key, resolution.model, item.prompt
        )
        _print_row(
            item.pool,
            item.prompt,
            resolution.model,
            latency_ms,
            cost_usd,
            response_ok,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
