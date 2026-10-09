# Theriad Architecture

## Overview

Theriad is a **cost-first AI inference optimization engine**.
It sits between the caller and the LLM provider and selects the cheapest model
that can produce acceptable quality output for each request.

## Routing pipeline (layers in priority order)

```
Prompt
  │
  ▼
┌─────────────────────────────────────┐
│  Layer 0 — Deterministic Rules      │  (regex, length, content type)
│  Fast, zero-cost, always runs first │
└────────────────┬────────────────────┘
                 │ confidence < threshold?
                 ▼
┌─────────────────────────────────────┐
│  Layer 1 — Feature Extraction       │  (numerical signals, no model)
│  token count, code ratio, etc.      │
└────────────────┬────────────────────┘
                 │ still uncertain?
                 ▼
┌─────────────────────────────────────┐
│  Layer 2 — Embedding Router         │  (local BERT/MiniLM, no API call)
│  pluggable embedding model          │
└────────────────┬────────────────────┘
                 │ confidence < threshold?
                 ▼
┌─────────────────────────────────────┐
│  Layer 3 — Optional Local Classifier│  (removable, isolated)
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Execution Engine                   │  retry / fallback / timeout
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Provider (OpenRouter / Ollama)     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Cost Accounting                    │  persist metrics to JSONL
└─────────────────────────────────────┘
```

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Python 3.10+, dataclasses for runtime types | Minimal overhead on hot path |
| Pydantic v2 for config only | Config parsed once; strong validation |
| YAML config | Human-friendly, ML-ecosystem standard |
| JSONL metrics store | Append-only, zero dependency, grep-able |
| Layers are pluggable ABCs | Add/remove without touching the engine |
| No RL, no auto-tuning | Out of scope for v1; simplicity first |

## Directory layout

```
theriad/
  core/
    config/       — schema + loader
    logging/      — structured JSON logger
    routing/      — RoutingRule, RoutingLayer, RouterProtocol ABCs
    providers/    — ProviderProtocol, CompletionRequest/Response
  layers/
    layer0/       — deterministic rules
    layer1_features/ — numerical feature extraction
    layer2_embedding/ — local embedding router
    layer3_classifier/ — optional local classifier (removable)
  engine/         — execution pipeline (retry, fallback, timeout)
  accounting/     — cost tracking + JSONL persistence
  cli/            — Click-based CLI (theriad route|benchmark|config|stats|explain)
tests/
  unit/           — fast, no I/O
  integration/    — requires provider access
```
